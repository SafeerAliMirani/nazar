"""Pull one MVTec AD category and lay it out the way repro.py expects.

Source is a third party mirror on HuggingFace that keeps the real images but
transposes the folders to images/<split>/<category>/<defect> with masks under
masks/. MVTec AD is CC BY-NC-SA 4.0 (Bergmann et al.). For anything public,
download the canonical archive from mvtec.com and attribute it.

Resumable and stubborn: files are written atomically, every file is retried,
and one dropped connection never kills the run. Safe to re-run any time.

    python get_leather.py --out D:\\data\\mvtec_ad --category screw
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "TheoM55/mvtec_anomaly_detection"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

UA = {"User-Agent": "nazar-p0"}


def target_for(rel, out):
    # images/train/leather/good/000.png     -> leather/train/good/000.png
    # images/test/leather/color/000.png     -> leather/test/color/000.png
    # masks/test/leather/color/000_mask.png -> leather/ground_truth/color/000_mask.png
    parts = rel.split("/")
    if parts[0] == "images" and len(parts) == 5:
        _, split, cat, defect, name = parts
        return out / cat / split / defect / name
    if parts[0] == "masks" and len(parts) == 5:
        _, _, cat, defect, name = parts
        return out / cat / "ground_truth" / defect / name
    return None


def have(dst):
    return dst.exists() and dst.stat().st_size > 0


def fetch(job):
    rel, dst = job
    if have(dst):
        return 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(RESOLVE + rel, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data:
                raise IOError("empty body")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dst)  # atomic, so a partial file never looks done
            return len(data)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return -1


def get_listing():
    for attempt in range(5):
        try:
            req = urllib.request.Request(API, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  listing attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    raise SystemExit("could not read the file listing, network is down")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default="leather")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=25)
    args = ap.parse_args()
    out = Path(args.out)

    meta = get_listing()
    jobs = []
    for s in meta["siblings"]:
        rel = s["rfilename"]
        if f"/{args.category}/" not in rel:
            continue
        dst = target_for(rel, out)
        if dst is not None:
            jobs.append((rel, dst))
    print(f"{len(jobs)} {args.category} files total", flush=True)

    for sweep in range(args.sweeps):
        todo = [j for j in jobs if not have(j[1])]
        if not todo:
            break
        print(f"sweep {sweep}: {len(todo)} missing", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(fetch, todo))
        done = sum(1 for j in jobs if have(j[1]))
        print(f"  -> {done}/{len(jobs)} present", flush=True)
        if [j for j in jobs if not have(j[1])]:
            time.sleep(2)

    root = out / args.category
    n_train = len(list((root / "train" / "good").glob("*.png")))
    n_test = len(list(root.glob("test/*/*.png")))
    n_mask = len(list(root.glob("ground_truth/*/*_mask.png")))
    n_good = len(list((root / "test" / "good").glob("*.png")))
    for p in root.rglob("*.part"):
        p.unlink()

    print(f"\ntrain/good     {n_train}")
    print(f"test total     {n_test}  (of which good {n_good})")
    print(f"ground_truth   {n_mask}")
    print(f"defect types   {sorted(d.name for d in (root / 'test').iterdir() if d.is_dir())}")
    ok = len([j for j in jobs if not have(j[1])]) == 0
    print("\nlayout complete, every listed file present" if ok
          else "\nINCOMPLETE, re-run this script, do not trust a number from partial data")
    print(f"data root for repro.py: {out}")


if __name__ == "__main__":
    main()
