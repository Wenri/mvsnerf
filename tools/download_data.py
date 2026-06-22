#!/usr/bin/env python3
"""Download the MVSNeRF datasets from the GitHub Release mirror into ds/.

Reconstructs the ds/ tree from assets published on the Wenri/mvsnerf releases,
described by tools/data_manifest.json. Standard library only (no pip installs).

Usage:
    python tools/download_data.py --list
    python tools/download_data.py --all
    python tools/download_data.py --dataset llff --dataset rs-dtu
    python tools/download_data.py --scene scan23          # only the parts holding scan23
    python tools/download_data.py --verify-only           # check cached archives vs sha256

Notes:
  * Training the generalizable DTU model (dataset_name dtu) needs BOTH
    --dataset mvs-training AND --dataset depths-raw (images come from
    mvs_training/.../Rectified/scanN_train, depths from ds/Depths/scanN).
  * Downloads resume (HTTP Range); archives are sha256-verified before extraction.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "data_manifest.json"
GIB = 1024 ** 3


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def load_manifest():
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    return json.loads(MANIFEST.read_text())


def asset_url(m, asset):
    return m["asset_url_template"].format(repo=m["repo"], tag=asset["tag"], asset=asset["name"])


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, size=None, retries=4):
    """Download url -> dest with resume + retry/backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if size is not None and have == size:
            return
        if size is not None and have > size:        # corrupt/stale partial
            dest.unlink()
            have = 0
        req = urllib.request.Request(url)
        mode = "wb"
        if have:
            req.add_header("Range", f"bytes={have}-")
            mode = "ab"
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and getattr(r, "status", 200) == 200:   # server ignored Range
                    mode = "wb"
                with open(dest, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    {type(e).__name__}: {e}; retry {attempt}/{retries - 1} in {wait}s")
            time.sleep(wait)


def safe_extract(zip_path, dest_root):
    dest_root = dest_root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (dest_root / name).resolve()
            if dest_root not in target.parents and target != dest_root:
                sys.exit(f"unsafe path in {zip_path.name}: {name}")
        zf.extractall(dest_root)


def select_assets(m, args):
    """Return ordered, de-duplicated list of asset dicts to fetch."""
    chosen, seen = [], set()

    def add(dataset_key, asset):
        k = (asset["tag"], asset["name"])
        if k not in seen:
            seen.add(k)
            chosen.append(asset)

    def add_dataset(key):
        d = m["datasets"][key]
        for a in d["assets"]:
            add(key, a)

    if args.all:
        for key in m["datasets"]:
            add_dataset(key)
    for key in args.dataset or []:
        if key not in m["datasets"]:
            sys.exit(f"unknown dataset {key}; choices: {', '.join(m['datasets'])}")
        add_dataset(key)
    for scene in args.scene or []:
        hits = m["scene_index"].get(scene)
        if not hits:
            print(f"  scene {scene} not found in manifest")
            continue
        for hit in hits:
            key = hit["dataset"]
            d = m["datasets"][key]
            by_name = {a["name"]: a for a in d["assets"]}
            add(key, by_name[hit["asset"]])
            cam = d.get("cameras_asset")            # a scene is useless without Cameras
            if cam:
                add(key, by_name[cam])
    return chosen


def cmd_list(m):
    print(f"repo: {m['repo']}   extract_root: {m['extract_root']}/")
    for key, d in m["datasets"].items():
        print(f"\n{key}  ({d['tag']})  ~{human(d['total_bytes'])}")
        print(f"  {d['description']}")
        for a in d["assets"]:
            print(f"    {a['name']:30s} {human(a['bytes']):>10s}  {a['format']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download every dataset")
    ap.add_argument("--dataset", action="append", help="dataset to download (repeatable)")
    ap.add_argument("--scene", action="append", help="download only parts holding this scene")
    ap.add_argument("--list", action="store_true", help="list datasets/assets and exit")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-hash cached archives against the manifest and exit")
    ap.add_argument("--dest", type=Path, default=ROOT / "ds",
                    help="extract root (default: ds/)")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep downloaded archives in <dest>/.download_cache")
    args = ap.parse_args()

    m = load_manifest()
    if args.list:
        return cmd_list(m)

    dest = args.dest
    cache = dest / ".download_cache"

    if args.verify_only:
        ok = bad = miss = 0
        for d in m["datasets"].values():
            for a in d["assets"]:
                f = cache / a["name"]
                if not f.exists():
                    miss += 1
                    continue
                good = sha256_file(f) == a["sha256"]
                print(f"  {'OK ' if good else 'BAD'} {a['name']}")
                ok, bad = ok + good, bad + (not good)
        print(f"verify: {ok} ok, {bad} bad, {miss} not cached")
        return sys.exit(1 if bad else 0)

    assets = select_assets(m, args)
    if not assets:
        ap.error("specify --all, --dataset NAME, --scene NAME, --list, or --verify-only")
    total = sum(a["bytes"] for a in assets)
    print(f"{len(assets)} asset(s), {human(total)} -> {dest}/")

    for a in assets:
        url, archive = asset_url(m, a), cache / a["name"]
        print(f"==> {a['name']} ({human(a['bytes'])})")
        download(url, archive, a["bytes"])
        digest = sha256_file(archive)
        if digest != a["sha256"]:
            archive.unlink(missing_ok=True)         # drop & retry once from scratch
            download(url, archive, a["bytes"])
            if sha256_file(archive) != a["sha256"]:
                sys.exit(f"sha256 mismatch for {a['name']}")
        if a["format"] == "file":                   # passthrough: place, don't extract
            (dest).mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, dest / a["name"])
        else:
            safe_extract(archive, dest)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
    if cache.exists() and not args.keep_archives and not any(cache.iterdir()):
        cache.rmdir()
    print("done.")


if __name__ == "__main__":
    main()
