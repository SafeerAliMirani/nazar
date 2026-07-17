"""Copy the shipped runs into web/app/data and rewrite its index.

This is the supervisor step. The web app never reads pipeline/runs directly, so
nothing reaches the page without passing through here. That is the point: a
number cannot appear on the site unless someone moved it deliberately.

Everything shipped is wide_resnet50_2 at 2 percent. The old resnet50 and 10
percent files are quarantined in artifacts/_stale and must not come back: the
banks are shape identical across backbones and no file announces which is which.

    python publish_to_app.py
"""

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
APP = HERE.parent / "web" / "app" / "data"
ART = HERE.parent / "artifacts"

CATS = ["screw", "capsule", "pill", "cable", "metal_nut", "transistor", "leather", "hazelnut"]
PRIMARY = "screw"          # the informative one. leather saturates and proves nothing.
WANT_BACKBONE = "wide_resnet50_2 IMAGENET1K_V1"
WANT_CORESET = 0.02


def score_keys(csv_path):
    # (defect_type, file), never file alone. MVTec reuses filenames across
    # defect types and a key of file alone quietly returns another image's row.
    keys = set()
    if not csv_path.is_file():
        return keys
    lines = csv_path.read_text().strip().splitlines()
    head = lines[0].split(",")
    for line in lines[1:]:
        if not line.strip():
            continue
        v = dict(zip(head, line.split(",")))
        keys.add((v["defect_type"], v["file"]))
    return keys


def parse_panel(name):
    # repro.py names these {rank:02d}_{dtype}_{stem}_score_{score:.4f}.png, and
    # dtype itself contains underscores, so take the stem off the right hand end.
    m = re.fullmatch(r"\d+_(.+)_score_-?[\d.]+\.png", name)
    if not m:
        return None
    dtype, _, stem = m.group(1).rpartition("_")
    if not dtype or not stem:
        return None
    return dtype, stem + ".png"


def main():
    APP.mkdir(parents=True, exist_ok=True)
    (APP / "heatmaps").mkdir(exist_ok=True)

    runs = []
    for cat in CATS:
        src = RUNS / f"ship_{cat}"
        mj = src / "metrics.json"
        if not mj.is_file():
            print(f"{cat}: no ship run, skipping")
            continue
        m = json.loads(mj.read_text())

        # refuse to publish anything that is not the shipped config. mixing a
        # backbone or a coreset size into the page is the exact failure the
        # sidecar warns about and nothing downstream would notice.
        if m["config"]["backbone"] != WANT_BACKBONE:
            raise SystemExit(f"{cat}: backbone is {m['config']['backbone']}, refusing")
        if abs(m["coreset_pct"] - WANT_CORESET) > 1e-9:
            raise SystemExit(f"{cat}: coreset is {m['coreset_pct']}, refusing")

        shutil.copy2(mj, APP / f"metrics_ship_{cat}.json")
        entry = {"id": f"ship_{cat}", "metrics": f"metrics_ship_{cat}.json",
                 "source": f"pipeline/runs/ship_{cat}"}
        sc = src / "scores.csv"
        if sc.is_file():
            shutil.copy2(sc, APP / f"scores_ship_{cat}.csv")
            entry["scores"] = f"scores_ship_{cat}.csv"

        hm = src / "heatmaps"
        if hm.is_dir():
            # the page reads png, defect_type and file off each entry. it used to
            # get file and src, so it asked for data/undefined, captioned every
            # image undefined, and missed every score row.
            keys = score_keys(sc)
            index = []
            for p in sorted(hm.glob("*.png")):
                dst = f"ship_{cat}_{p.name}"
                shutil.copy2(p, APP / "heatmaps" / dst)
                parsed = parse_panel(p.name)
                if parsed is None:
                    raise SystemExit(f"{cat}: cannot read a defect type out of {p.name}")
                dtype, fname = parsed
                # the caption number is joined on this key downstream and a miss
                # there prints "no score row" rather than failing, so check here.
                if keys and (dtype, fname) not in keys:
                    raise SystemExit(
                        f"{cat}: {p.name} parses to {dtype}/{fname}, which is not a "
                        f"row in scores.csv. do not ship this index.")
                index.append({"png": f"heatmaps/{dst}", "defect_type": dtype,
                              "file": fname, "src": p.name})
            if index:
                (APP / f"heatmaps_ship_{cat}.json").write_text(json.dumps(index, indent=2))
                entry["heatmaps"] = f"heatmaps_ship_{cat}.json"

        runs.append(entry)
        print(f"{cat}: published, auroc_plain {m['image_auroc_plain']:.4f}, bank {m['bank_size']}")

    # The ablated leather run is the evidence that a good score on an easy
    # category proves nothing. It has to be the SHIPPED config, otherwise it is
    # not comparable to the clean run and the page is right to refuse to pair
    # them. The old resnet50 10 percent ablation is not published for that reason.
    abl = RUNS / "ship_leather_ablated" / "metrics.json"
    if abl.is_file():
        m = json.loads(abl.read_text())
        if m["config"]["backbone"] != WANT_BACKBONE or abs(m["coreset_pct"] - WANT_CORESET) > 1e-9:
            raise SystemExit("the ablation is not the shipped config, refusing to publish it")
        shutil.copy2(abl, APP / "metrics_ship_leather_ablated.json")
        runs.append({"id": "ship_leather_ablated",
                     "metrics": "metrics_ship_leather_ablated.json",
                     "source": "pipeline/runs/ship_leather_ablated",
                     "note": "bank deliberately destroyed, kept as evidence"})
        print(f"leather ablation published, auroc_plain {m['image_auroc_plain']:.4f} "
              f"against a clean 1.0000")

    for f in ["categories.json", "compression.json", "curve.json"]:
        if (ART / f).is_file():
            shutil.copy2(ART / f, APP / f)

    sidecar = f"sidecar_{PRIMARY}_wide_resnet50_2_c2.json"
    if (ART / sidecar).is_file():
        shutil.copy2(ART / sidecar, APP / sidecar)

    (APP / "index.json").write_text(json.dumps({
        "note": ("File list only. Every label and figure the page shows is read out of "
                 "the files named here, never out of this manifest. Copied from "
                 "pipeline/runs by the supervisor step, publish_to_app.py."),
        "shipped_config": {"backbone": WANT_BACKBONE, "coreset_pct": WANT_CORESET},
        "primary_run": f"ship_{PRIMARY}",
        "runs": runs,
        "curve": "curve.json",
        "categories": "categories.json",
        "compression": "compression.json",
        "sidecar": sidecar,
    }, indent=2))

    # drop anything left over from the resnet50 / 10 percent era
    for p in list(APP.glob("*_c10.json")) + list(APP.glob("*screw_both*")) + \
             list(APP.glob("*screw_wrn*")) + list(APP.glob("*screw_bal*")) + \
             list(APP.glob("*leather_export*")) + list(APP.glob("metrics_ablate_shuffle.json")):
        p.unlink()
        print(f"removed stale {p.name}")

    print(f"\nprimary_run: ship_{PRIMARY}, {len(runs)} runs published, all "
          f"{WANT_BACKBONE} at {WANT_CORESET:.0%}")


if __name__ == "__main__":
    main()
