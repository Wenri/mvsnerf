#!/usr/bin/env python3
"""Package the datasets under ds/ and publish them as GitHub Release assets.

Grouping rule: one archive per dataset when it fits under the asset ceiling,
otherwise bundle WHOLE SCENES into the fewest <=1.95 GiB parts (never splitting a
file across parts). Archives are zip-store with members stored relative to ds/,
so the download helper just extracts them into ds/.

Idempotent + resumable: existing releases are reused, already-uploaded assets are
skipped, and uploads use --clobber so a re-run after a failure is safe.

Requires the `gh` CLI (authenticated). Operates on the gitignored ds/ tree; the
script itself is committed for reproducibility.

Usage:
    python tools/publish_datasets.py --dry-run            # plan everything, no network
    python tools/publish_datasets.py --dataset rs-dtu     # publish one dataset
    python tools/publish_datasets.py --all                # publish everything
    python tools/publish_datasets.py --manifest-only      # (re)write tools/data_manifest.json
    python tools/publish_datasets.py --clean              # remove the staging dir
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "Wenri/mvsnerf"
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "ds"
STAGE = DS / ".publish_stage"          # under ds/ -> covered by the existing gitignore
CACHE = STAGE / ".uploaded.json"
MANIFEST = ROOT / "tools" / "data_manifest.json"

GIB = 1024 ** 3
BIN_LIMIT = int(1.95 * GIB)            # packing ceiling (margin under the 2 GiB hard cap)
HARD_LIMIT = 2 * GIB                   # GitHub per-asset hard limit
ENTRY_OVERHEAD = 64                    # conservative per-file zip header estimate (bytes)
ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"


# --------------------------------------------------------------------------- #
# unit discovery
# --------------------------------------------------------------------------- #
# A "unit" is the indivisible packing element (one scene, or the shared cameras
# blob). It is a dict: {name, scenes:[...], size:int, files:[(arc, size, src)]}
# where src is ("fs", abspath) or ("zip", member_name).

def _iter_files(d):
    d = Path(d)
    if not d.is_dir():
        return
    for r, _, fs in os.walk(d):
        for f in fs:
            yield Path(r) / f


def _mk_unit(name, entries, scenes):
    return {"name": name, "scenes": scenes, "files": entries,
            "size": sum(sz + ENTRY_OVERHEAD for _, sz, _ in entries)}


def _matcher(pattern):
    pat = re.compile(pattern)

    def scene_of(arc):
        m = pat.match(arc)
        return m.group(1) if m else None
    return scene_of


def _discover_tree(root, scene_of):
    """Walk EVERY file under root; scene_of(arcname) -> scene name or None.

    Completeness-guaranteed: each file under root is assigned to exactly one
    scene unit or the shared 'cameras' unit, so no file is ever dropped.
    """
    scene_files, shared = {}, []
    for p in _iter_files(root):
        entry = (str(p.relative_to(DS)), p.stat().st_size, ("fs", str(p)))
        s = scene_of(entry[0])
        if s is None:
            shared.append(entry)
        else:
            scene_files.setdefault(s, []).append(entry)
    units = [_mk_unit(s, scene_files[s], [s]) for s in sorted(scene_files)]
    shared_units = [_mk_unit("cameras", shared, [])] if shared else []
    return units, shared_units


def discover_mvs_training():
    return _discover_tree(
        DS / "mvs_training",
        _matcher(r"mvs_training/dtu/(?:Rectified|Depths)/(scan\d+_train)/"))


def discover_depths_raw():
    """Re-pack straight from Depths_raw.zip (no extraction)."""
    zpath = DS / "Depths_raw.zip"
    by_scene, shared = {}, []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            entry = (info.filename, info.file_size, ("zip", info.filename))
            parts = info.filename.split("/")
            if len(parts) >= 3 and parts[0] == "Depths":
                by_scene.setdefault(parts[1], []).append(entry)
            else:
                shared.append(entry)
    units = [_mk_unit(s, by_scene[s], [s]) for s in sorted(by_scene)]
    shared_units = [_mk_unit("misc", shared, [])] if shared else []
    return units, shared_units


def discover_llff():
    return _discover_tree(DS / "llff", _matcher(r"llff/nerf_llff_data/([^/]+)/"))


def discover_rs_dtu():
    return _discover_tree(DS / "rs_dtu", _matcher(r"rs_dtu/DTU/(scan\d+)/"))


def discover_dtu():
    return _discover_tree(
        DS / "dtu", _matcher(r"dtu/(?:Rectified|Depths)/(scan\d+(?:_train)?)/"))


DATASETS = {
    "mvs-training": dict(
        tag="data-mvs-training", prefix="mvs_training", path_root="mvs_training/dtu",
        discover=discover_mvs_training, source_zip=None,
        title="DTU training set (Rectified + Depths + Cameras)",
        description="DTU preprocessed training images and depths "
                    "(Rectified/scanN_train, Depths/scanN_train) + shared Cameras.",
        provenance="DTU training data, Google Drive (MVSNeRF preprocessing of MVSNet "
                   "https://github.com/YoYo000/MVSNet). NOTE: the generalization 'dtu' "
                   "loader also needs plain Depths/scanN from the 'depths-raw' dataset."),
    "depths-raw": dict(
        tag="data-depths-raw", prefix="depths_raw", path_root="Depths",
        discover=discover_depths_raw, source_zip=str(DS / "Depths_raw.zip"),
        title="DTU raw depth maps (Depths/)",
        description="CasMVSNet raw depth maps; extracts to ds/Depths/ "
                    "(scanN and scanN_train).",
        provenance="Depths_raw.zip from CasMVSNet (aliyun OSS: virutalbuy-public..."
                   "/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)."),
    "llff": dict(
        tag="data-llff", prefix="llff", path_root="llff/nerf_llff_data",
        discover=discover_llff, source_zip=None,
        title="LLFF forward-facing scenes",
        description="nerf_llff_data: fern, flower, fortress, horns, leaves, orchids, "
                    "room, trex.",
        provenance="nerf_llff_data (Google Drive, original NeRF/LLFF release)."),
    "rs-dtu": dict(
        tag="data-rs-dtu", prefix="rs_dtu", path_root="rs_dtu/DTU",
        discover=discover_rs_dtu, source_zip=None,
        passthrough=["dtu_example.zip"],
        title="rs_dtu test set (+ dtu_example.zip)",
        description="rs_dtu/DTU test scans + Cameras. Includes dtu_example.zip "
                    "(restored as the file ds/dtu_example.zip, not extracted).",
        provenance="rs_dtu (MVSNeRF). dtu_example.zip from Hugging Face apchen/MVSNeRF."),
    "dtu": dict(
        tag="data-dtu", prefix="dtu", path_root="dtu",
        discover=discover_dtu, source_zip=None,
        title="DTU example subset (3 scenes)",
        description="Small DTU subset (Rectified: scan31, scan31_train, scan114; "
                    "Depths: scan114) for smoke tests.",
        provenance="Subset assembled from DTU preprocessed data."),
}


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
def ffd_pack(units, limit=BIN_LIMIT):
    """First-fit-decreasing; deterministic (size desc, then name)."""
    bins = []
    for u in sorted(units, key=lambda u: (-u["size"], u["name"])):
        if u["size"] > limit:
            sys.exit(f"unit {u['name']} ({u['size']}B) exceeds bin limit {limit}B")
        placed = next((b for b in bins if b["size"] + u["size"] <= limit), None)
        if placed is None:
            placed = {"size": 0, "units": []}
            bins.append(placed)
        placed["units"].append(u)
        placed["size"] += u["size"]
    return bins


def plan_assets(cfg, scene_units, shared):
    """Return a list of asset dicts: {name, units, scenes, path_root}."""
    prefix, root = cfg["prefix"], cfg["path_root"]
    bins = ffd_pack(scene_units + shared)
    if len(bins) <= 1:                       # whole dataset fits one asset
        units = bins[0]["units"] if bins else []
        scenes = sorted({s for u in units for s in u["scenes"]})
        return [dict(name=f"{prefix}.zip", units=units, scenes=scenes, path_root=root)]
    out = []
    if shared:                               # multi-part -> cameras gets its own asset
        out.append(dict(name=f"{prefix}_cameras.zip", units=shared,
                        scenes=[], path_root=root))
    for i, b in enumerate(ffd_pack(scene_units), 1):
        scenes = sorted({s for u in b["units"] for s in u["scenes"]})
        out.append(dict(name=f"{prefix}_part{i:02d}.zip", units=b["units"],
                        scenes=scenes, path_root=root))
    return out


# --------------------------------------------------------------------------- #
# building + gh
# --------------------------------------------------------------------------- #
def build_zip(zip_path, units, source_zip=None):
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    src = zipfile.ZipFile(source_zip) if source_zip else None
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for u in units:
                for arc, _, (kind, ref) in sorted(u["files"]):
                    if kind == "fs":
                        zf.write(ref, arc)
                    else:
                        zf.writestr(arc, src.read(ref))
    finally:
        if src:
            src.close()
    size = tmp.stat().st_size
    if size >= HARD_LIMIT:
        tmp.unlink()
        sys.exit(f"{zip_path.name} is {size}B >= 2 GiB hard limit; lower BIN_LIMIT")
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    tmp.rename(zip_path)
    return size, h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gh(*args, check=True, capture=False):
    return subprocess.run(["gh", *args], check=check, text=True,
                          capture_output=capture)


def ensure_release(tag, title, notes):
    r = gh("release", "view", tag, "-R", REPO, check=False, capture=True)
    if r.returncode != 0:
        gh("release", "create", tag, "-R", REPO, "-t", title, "-n", notes)
        print(f"  created release {tag}")


def remote_assets(tag):
    r = gh("release", "view", tag, "-R", REPO, "--json", "assets",
           check=False, capture=True)
    if r.returncode != 0:
        return {}
    return {a["name"]: a["size"] for a in json.loads(r.stdout).get("assets", [])}


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(cache):
    STAGE.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #
def publish_dataset(key, dry_run=False):
    cfg = DATASETS[key]
    print(f"== {key} ({cfg['tag']}) ==")
    scene_units, shared = cfg["discover"]()
    assets = plan_assets(cfg, scene_units, shared)
    total = sum(u["size"] for a in assets for u in a["units"])
    nfiles = sum(len(u["files"]) for a in assets for u in a["units"])
    n_scenes = len({s for a in assets for s in a["scenes"]})
    print(f"  {len(scene_units)} scenes, {nfiles} files -> {len(assets)} asset(s), "
          f"~{total/GIB:.2f} GiB ({n_scenes} scenes labelled)")
    for a in assets:
        est = sum(u["size"] for u in a["units"]) / GIB
        print(f"    {a['name']:32s} ~{est:5.2f} GiB  {len(a['scenes'])} scenes")
    if cfg.get("passthrough"):
        for f in cfg["passthrough"]:
            print(f"    {f:32s} {(DS / f).stat().st_size/GIB:5.2f} GiB  (passthrough)")
    if dry_run:
        return

    STAGE.mkdir(parents=True, exist_ok=True)
    ensure_release(cfg["tag"], cfg["title"],
                   cfg["description"] + "\n\nProvenance: " + cfg["provenance"])
    remote = remote_assets(cfg["tag"])
    cache = load_cache()

    for a in assets:
        name = a["name"]
        if name in remote and cache.get(name, {}).get("size") == remote[name]:
            print(f"  skip {name} (uploaded)")
            continue
        zp = STAGE / name
        print(f"  building {name} ...")
        size, sha = build_zip(zp, a["units"], cfg["source_zip"])
        print(f"  uploading {name} ({size/GIB:.2f} GiB) ...")
        gh("release", "upload", cfg["tag"], str(zp), "-R", REPO, "--clobber")
        cache[name] = dict(tag=cfg["tag"], size=size, sha256=sha, format="zip-store",
                           path_root=a["path_root"], scenes=a["scenes"])
        save_cache(cache)
        zp.unlink()

    for f in cfg.get("passthrough", []):
        src = DS / f
        if f in remote and cache.get(f, {}).get("size") == remote[f]:
            print(f"  skip {f} (uploaded)")
            continue
        print(f"  uploading {f} (passthrough) ...")
        gh("release", "upload", cfg["tag"], str(src), "-R", REPO, "--clobber")
        cache[f] = dict(tag=cfg["tag"], size=src.stat().st_size, sha256=sha256_file(src),
                        format="file", path_root="", scenes=[])
        save_cache(cache)


def write_manifest():
    cache = load_cache()
    tag2key = {c["tag"]: k for k, c in DATASETS.items()}
    datasets, scene_index = {}, {}
    for asset, info in sorted(cache.items()):
        key = tag2key.get(info["tag"])
        if key is None:
            continue
        d = datasets.setdefault(key, dict(
            tag=info["tag"], description=DATASETS[key]["description"],
            provenance=DATASETS[key]["provenance"], total_bytes=0, assets=[]))
        d["total_bytes"] += info["size"]
        d["assets"].append(dict(name=asset, tag=info["tag"], bytes=info["size"],
                                sha256=info["sha256"], format=info["format"],
                                path_root=info["path_root"], scenes=info["scenes"]))
        for s in info["scenes"]:
            scene_index.setdefault(s, []).append(dict(dataset=key, asset=asset))
    for d in datasets.values():
        cams = [a["name"] for a in d["assets"] if a["name"].endswith("_cameras.zip")]
        if cams:
            d["cameras_asset"] = cams[0]
    manifest = dict(schema_version=1, repo=REPO, asset_url_template=ASSET_URL,
                    extract_root="ds",
                    generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    datasets=datasets, scene_index=scene_index)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST} ({len(cache)} assets, {len(scene_index)} scenes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(DATASETS),
                    help="dataset to publish (repeatable)")
    ap.add_argument("--all", action="store_true", help="publish every dataset")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no network")
    ap.add_argument("--manifest-only", action="store_true",
                    help="(re)write the manifest from the upload cache")
    ap.add_argument("--clean", action="store_true", help="remove the staging dir and exit")
    args = ap.parse_args()

    if args.clean:
        import shutil
        if STAGE.exists():
            shutil.rmtree(STAGE)
        print(f"removed {STAGE}")
        return
    if args.manifest_only:
        write_manifest()
        return

    # publish smallest-first so the long pole (depths-raw) runs last
    order = ["rs-dtu", "dtu", "llff", "mvs-training", "depths-raw"]
    if args.all:
        targets = order
    elif args.dataset:
        targets = [k for k in order if k in args.dataset]
    else:
        ap.error("specify --all, --dataset NAME, --dry-run with a dataset, "
                 "--manifest-only, or --clean")

    for key in targets:
        publish_dataset(key, dry_run=args.dry_run)
    if not args.dry_run:
        write_manifest()


if __name__ == "__main__":
    main()
