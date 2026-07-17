"""Turn raw runs into the two data files the web page reads.

published numbers are transcribed from the PatchCore CVPR 2022 supplementary,
Table S1 (image AUROC) and Table S2 (pixelwise AUROC):
https://openaccess.thecvf.com/content/CVPR2022/supplemental/Roth_Towards_Total_Recall_CVPR_2022_supplemental.pdf

The paper's PatchCore-1 / -10 / -25 are coreset percentages, so they line up
with our --coreset-pct axis directly. Only 1, 10 and 25 exist in the table;
2 and 5 have no published counterpart and stay null.

    python report.py --compression      # ../artifacts/compression.json
    python report.py --categories       # ../artifacts/categories.json
"""

import argparse
import json
from pathlib import Path

SOURCE = ("Roth et al., Towards Total Recall in Industrial Anomaly Detection, "
          "CVPR 2022 supplementary, Table S1 (image AUROC) and Table S2 "
          "(pixel AUROC). WideResNet50 backbone, 224x224, PatchCore-{pct}.")

# Table S1. keyed by coreset percent, then category.
PUBLISHED_IMAGE = {
    1: {"bottle": 100.0, "cable": 99.3, "capsule": 98.0, "carpet": 98.0,
        "grid": 98.6, "hazelnut": 100.0, "leather": 100.0, "metal_nut": 99.7,
        "pill": 97.0, "screw": 96.4, "tile": 99.4, "toothbrush": 100.0,
        "transistor": 99.9, "wood": 99.2, "zipper": 99.2},
    10: {"bottle": 100.0, "cable": 99.4, "capsule": 97.8, "carpet": 98.7,
         "grid": 97.9, "hazelnut": 100.0, "leather": 100.0, "metal_nut": 100.0,
         "pill": 96.0, "screw": 97.0, "tile": 98.9, "toothbrush": 99.7,
         "transistor": 100.0, "wood": 99.0, "zipper": 99.5},
    25: {"bottle": 100.0, "cable": 99.5, "capsule": 98.1, "carpet": 98.7,
         "grid": 98.2, "hazelnut": 100.0, "leather": 100.0, "metal_nut": 100.0,
         "pill": 96.6, "screw": 98.1, "tile": 98.7, "toothbrush": 100.0,
         "transistor": 100.0, "wood": 99.2, "zipper": 99.4},
}

# Table S2.
PUBLISHED_PIXEL = {
    1: {"bottle": 98.5, "cable": 98.2, "capsule": 98.8, "carpet": 98.9,
        "grid": 98.6, "hazelnut": 98.6, "leather": 99.3, "metal_nut": 98.4,
        "pill": 97.1, "screw": 99.2, "tile": 96.1, "toothbrush": 98.5,
        "transistor": 94.9, "wood": 95.1, "zipper": 98.8},
    10: {"bottle": 98.6, "cable": 98.5, "capsule": 98.9, "carpet": 99.1,
         "grid": 98.7, "hazelnut": 98.7, "leather": 99.3, "metal_nut": 98.4,
         "pill": 97.6, "screw": 99.4, "tile": 95.9, "toothbrush": 98.7,
         "transistor": 96.4, "wood": 95.1, "zipper": 98.9},
    25: {"bottle": 98.6, "cable": 98.4, "capsule": 98.8, "carpet": 99.0,
         "grid": 98.7, "hazelnut": 98.7, "leather": 99.3, "metal_nut": 98.4,
         "pill": 97.4, "screw": 99.4, "tile": 95.6, "toothbrush": 98.7,
         "transistor": 96.3, "wood": 95.0, "zipper": 98.8},
}

DIMS = 1536


def published(table, category, pct):
    # only the three percentages the paper actually reports
    key = round(pct * 100)
    if abs(pct * 100 - key) > 1e-9 or key not in table:
        return None
    return table[key].get(category)


def source_for(pct):
    key = round(pct * 100)
    return SOURCE.format(pct=key) if key in PUBLISHED_IMAGE else None


def build_compression(raw_path, out_path):
    raw = json.loads(Path(raw_path).read_text())
    recs = []
    for r in raw:
        pct = float(r["coreset_pct"])
        rows = int(r["bank_rows"])
        recs.append({
            "category": r["category"],
            "backbone": r["backbone"],
            "coreset_pct": pct,
            "bank_rows": rows,
            "bank_bytes_f16": rows * DIMS * 2,
            "image_auroc_plain": float(r["image_auroc_plain"]),
            "image_auroc_eq7": float(r["image_auroc_eq7"]),
            "pixel_auroc": float(r["pixel_auroc"]),
            "published_image_auroc": published(PUBLISHED_IMAGE, r["category"], pct),
            "published_pixel_auroc": published(PUBLISHED_PIXEL, r["category"], pct),
            "source": source_for(pct),
        })
    recs.sort(key=lambda r: (r["category"], r["coreset_pct"]))
    Path(out_path).write_text(json.dumps(recs, indent=2))
    print(f"wrote {out_path} with {len(recs)} records")
    for r in recs:
        pub = r["published_image_auroc"]
        pub_s = f"{pub:.1f}" if pub is not None else "  - "
        print(f"  {r['category']:<11} pct={r['coreset_pct']:<5} rows={r['bank_rows']:>6} "
              f"{r['bank_bytes_f16']/1e6:>6.1f} MB  plain={r['image_auroc_plain']:.4f} "
              f"eq7={r['image_auroc_eq7']:.4f} pixel={r['pixel_auroc']:.4f}  pub={pub_s}")


def paper_comparison(raw_path, category, at_pct=0.10):
    """Our number at a percentage the paper actually reports.

    The shipped bank is 2%, which the paper never published, so the shipped row
    on its own cannot be compared to anything. This carries the same-pct run so
    the page can show a real comparison instead of a null.
    """
    raw = json.loads(Path(raw_path).read_text())
    for r in raw:
        if r["category"] == category and abs(float(r["coreset_pct"]) - at_pct) < 1e-9:
            pub = published(PUBLISHED_IMAGE, category, at_pct)
            ours = float(r["image_auroc_plain"]) * 100
            return {
                "coreset_pct": at_pct,
                "bank_rows": int(r["bank_rows"]),
                "image_auroc_plain": float(r["image_auroc_plain"]),
                "pixel_auroc": float(r["pixel_auroc"]),
                "published_image_auroc": pub,
                "published_pixel_auroc": published(PUBLISHED_PIXEL, category, at_pct),
                "gap_points": round(ours - pub, 3) if pub is not None else None,
                "source": source_for(at_pct),
            }
    return None


def build_categories(runs_dir, out_path, raw_path, prefix="ship_"):
    recs = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        m = json.loads((d / "metrics.json").read_text())
        cat = m["category"]
        pct = float(m["coreset_pct"])
        rows = int(m["bank_size"])
        counts = m["test_counts"]
        assert m["ablate_bank"] == "none", f"{d.name} is an ablation, not a real run"
        assert not m["permute_labels"], f"{d.name} has permuted labels"
        recs.append({
            "category": cat,
            "backbone": m["config"]["backbone"].split()[0],
            "coreset_pct": pct,
            "bank_rows": rows,
            "bank_bytes_f16": rows * DIMS * 2,
            "n_train": int(m["n_train"]),
            "n_test_good": int(m["n_test_good"]),
            "n_test_defect": int(m["n_test_defect"]),
            "defect_types": sorted(t for t in counts if t != "good"),
            "image_auroc_plain": float(m["image_auroc_plain"]),
            "image_auroc_eq7": float(m["image_auroc_eq7"]),
            "pixel_auroc": float(m["pixel_auroc"]),
            # the paper publishes 1, 10 and 25 only, so a 2% bank has no
            # published counterpart. null, not a guess.
            "published_image_auroc": published(PUBLISHED_IMAGE, cat, pct),
            "published_pixel_auroc": published(PUBLISHED_PIXEL, cat, pct),
            "source": source_for(pct),
            "paper_comparison": paper_comparison(raw_path, cat),
        })
    recs.sort(key=lambda r: r["category"])
    Path(out_path).write_text(json.dumps(recs, indent=2))
    print(f"wrote {out_path} with {len(recs)} categories\n")
    hdr = (f"{'category':<11} {'shipped':>8} {'MB':>6} | {'ours@10':>8} {'pub@10':>7} "
           f"{'gap':>7} | {'pixel@2':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        c = r["paper_comparison"]
        gap = c["gap_points"] if c else None
        flag = " <-- gap > 1.5" if gap is not None and abs(gap) > 1.5 else ""
        print(f"{r['category']:<11} {r['image_auroc_plain']*100:8.2f} "
              f"{r['bank_bytes_f16']/1e6:6.1f} | {c['image_auroc_plain']*100:8.2f} "
              f"{c['published_image_auroc']:7.1f} {gap:+7.2f} | "
              f"{r['pixel_auroc']*100:8.2f}{flag}")
    n = len(recs)
    print(f"\nmean shipped@2% {sum(r['image_auroc_plain'] for r in recs)/n*100:.3f}   "
          f"mean ours@10% {sum(r['paper_comparison']['image_auroc_plain'] for r in recs)/n*100:.3f}   "
          f"mean published@10% {sum(r['paper_comparison']['published_image_auroc'] for r in recs)/n:.3f}")
    print(f"total shipped payload all {n} banks: "
          f"{sum(r['bank_bytes_f16'] for r in recs)/1e6:.1f} MB f16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compression", action="store_true")
    ap.add_argument("--categories", action="store_true")
    ap.add_argument("--raw", default="../artifacts/compression_raw.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", default="../artifacts")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.compression:
        build_compression(args.raw, out / "compression.json")
    if args.categories:
        build_categories(args.runs, out / "categories.json", args.raw)


if __name__ == "__main__":
    main()
