"""Rebuild the per-category live query indexes without re-running the GPU.

The old export wrote queries/index.json under a fixed name while the banks and
the feature files were category tagged, so exporting leather clobbered screw's
index. The feature files themselves survived, and every reference score is in
the matching runs/<run>/scores.csv, so the indexes can be reconstructed from
what is already on disk.

    python rebuild_query_index.py
"""

import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts" / "queries"

# tag -> the run whose scores.csv holds the reference numbers for those images
SETS = {
    "screw_wide_resnet50_2": {"category": "screw", "run": "screw_live"},
    "leather_wide_resnet50_2": {"category": "leather", "run": "leather_wrn"},
}


def load_scores(run):
    # keyed by (defect_type, file). MVTec reuses filenames across defect types,
    # so file alone collides and quietly hands back another image's score.
    rows = {}
    csv = HERE / "runs" / run / "scores.csv"
    if not csv.is_file():
        return rows
    lines = csv.read_text().strip().splitlines()
    head = lines[0].split(",")
    for line in lines[1:]:
        v = dict(zip(head, line.split(",")))
        rows[(v["defect_type"], v["file"])] = v
    return rows


def main():
    sets = []
    for tag, spec in SETS.items():
        cat = spec["category"]
        scores = load_scores(spec["run"])
        if not scores:
            print(f"{tag}: no scores.csv for run {spec['run']}, skipping")
            continue

        images = []
        for f in sorted(ART.glob(f"{cat}_*.f16.bin")):
            name = f.name[: -len(".f16.bin")]
            # name is <category>_<defect_type>_<stem>
            rest = name[len(cat) + 1:]
            stem = rest.rsplit("_", 1)[-1]
            dtype = rest[: -(len(stem) + 1)]
            row = scores.get((dtype, f"{stem}.png"))
            if row is None:
                print(f"  {name}: no score row, skipping")
                continue
            png = ART / f"{name}.png"
            if not png.is_file():
                print(f"  {name}: no png, skipping")
                continue
            w = float(row["w"])
            sc = float(row["score_eq7"])
            images.append({
                "name": name, "label": int(row["label"]), "defect_type": dtype,
                "file": f"{stem}.png", "patches": 784, "dims": 1536, "grid": [28, 28],
                "reference": {"score_eq7": sc, "w": w, "d_star": float(row["d_star"])},
            })

        if not images:
            print(f"{tag}: no images found, skipping")
            continue

        idx = {
            "category": cat, "backbone": "wide_resnet50_2",
            "bank": f"bank_{tag}_c10", "b_nearest": 9, "crop": 224, "blur_sigma": 4.0,
            "note": ("patch features are precomputed offline. the kNN, the eq.7 "
                     "reweighting and the heatmap run in WGSL on your GPU."),
            "rebuilt_from": f"runs/{spec['run']}/scores.csv",
            "images": images,
        }
        # the reference score is the only reason this file exists, so check the
        # join rather than assume it: every image must have its own numbers
        seen = {}
        for im in images:
            k = round(im["reference"]["score_eq7"], 6)
            if k in seen:
                raise SystemExit(
                    f"{tag}: {im['name']} and {seen[k]} share score {k}. the join "
                    f"is collapsing rows, do not ship these references.")
            seen[k] = im["name"]
        for im in images:
            want = 0 if im["defect_type"] == "good" else 1
            if im["label"] != want:
                raise SystemExit(
                    f"{tag}: {im['name']} is defect_type={im['defect_type']} but "
                    f"label={im['label']}. the join is wrong.")

        (ART / f"{tag}.json").write_text(json.dumps(idx, indent=2))
        sets.append({"tag": tag, "category": cat, "backbone": "wide_resnet50_2",
                     "index": f"{tag}.json", "bank": f"bank_{tag}_c10",
                     "images": len(images)})
        print(f"{tag}: wrote {tag}.json with {len(images)} images")

    (ART / "manifest.json").write_text(json.dumps({
        "note": "each set is a separate bank download. switching costs bandwidth.",
        "sets": sorted(sets, key=lambda s: s["tag"])}, indent=2))
    print(f"\nmanifest lists: {', '.join(s['tag'] for s in sets)}")
    old = ART / "index.json"
    if old.is_file():
        old.unlink()
        print("removed the old fixed-name index.json that caused the clobber")


if __name__ == "__main__":
    main()
