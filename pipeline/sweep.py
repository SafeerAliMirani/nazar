"""Few-shot curve: how many good parts does PatchCore need before it works?

Sweeps category x backbone x n_shot x seed at a fixed 10% coreset and writes
../artifacts/curve.json. Every piece of feature extraction, coreset selection
and scoring is imported from repro.py, the code that passed the P0 gate, so
this file cannot drift away from the number that was verified.

    python sweep.py --data-root D:\\data\\mvtec_ad

Resumable: grid points already present in curve.json are skipped.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import repro

SHOTS = [1, 2, 5, 10, 25]
SEEDS = [0, 1, 2]
FULL_SEED = 0

RECORD_KEYS = ("category", "backbone", "n_shot", "seed", "coreset_pct")


def quiet_tqdm(it, **kwargs):
    return it


def train_subset(train_paths, n, seed):
    """n training images drawn for this seed.

    One permutation per seed, then take the first n, so the subsets are nested
    inside a seed: the 1-shot image is also in the 2-shot set. That keeps the
    climb attributable to n instead of to which images happened to be drawn.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_paths))
    return [train_paths[i] for i in sorted(order[:n].tolist())]


def grid_for(train_paths):
    pts = [(n, s) for n in SHOTS if n < len(train_paths) for s in SEEDS]
    # full train set: the selection is deterministic given a fixed init, there
    # is no draw left to randomise, so one seed is the whole distribution.
    pts.append((len(train_paths), FULL_SEED))
    return pts


def point_key(rec):
    return (rec["category"], rec["backbone"], int(rec["n_shot"]),
            int(rec["seed"]), round(float(rec["coreset_pct"]), 6))


def load_curve(path):
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def run_point(category, backbone, n_shot, seed, pcts, model, taps,
              train_paths, test_items, labels, masks, device):
    repro.seed_everything(seed)
    paths = train_subset(train_paths, n_shot, seed)
    assert len(paths) == n_shot, f"asked for {n_shot} images, drew {len(paths)}"
    assert len(set(paths)) == n_shot, "the same training image was drawn twice"
    train_set = set(train_paths)
    assert all(p in train_set for p in paths), "a k-shot image came from outside train/good"

    feats, grid = repro.extract_train(paths, model, taps, device)
    n_patch = grid[0] * grid[1]
    # REVIEW_V2 item 14. If n images are sampled but the bank is built from all
    # 245 anyway, the curve comes out flat and perfect and the curve is the
    # entire product. This is the assertion that catches it.
    assert feats.shape[0] == n_shot * n_patch, (
        f"features are {feats.shape[0]} rows, expected {n_shot} images x "
        f"{n_patch} patches = {n_shot * n_patch}. The bank is not being built "
        f"from only the {n_shot} sampled images.")

    # greedy k-center is prefix-nested: the first m picks at budget k_max are
    # identical to the picks at budget m. Select once at the largest budget and
    # slice, the k axis is free.
    pct_max = max(pcts)
    assert pct_max < 1.0, "pct 1.0 short circuits to arange and is not a greedy order"
    picked, proj_dim = repro.coreset_indices(feats, pct_max, seed, device)

    out = []
    for pct in sorted(pcts):
        m = max(1, int(round(feats.shape[0] * pct)))
        idx = picked[:m]
        assert len(set(idx.tolist())) == m, "the coreset prefix contains a duplicate"
        bank = feats[torch.from_numpy(idx)].clone()
        expect = max(1, int(round(n_shot * n_patch * pct)))
        assert bank.shape[0] == expect, (
            f"bank is {bank.shape[0]} rows, expected {n_shot} x {n_patch} x "
            f"{pct} = {expect}")

        scores, maps, ws = repro.score_test(test_items, model, taps, bank, device)
        # scores = w * d_star. The official PatchCore scores with d_star alone,
        # so the plain number is the one comparable to the published table.
        d_star = scores / np.maximum(ws, 1e-12)
        assert maps.shape == masks.shape, (
            f"maps {maps.shape} and masks {masks.shape} must match at metric time")
        out.append({
            "category": category,
            "backbone": backbone,
            "n_shot": int(n_shot),
            "seed": int(seed),
            "coreset_pct": float(pct),
            "bank_rows": int(bank.shape[0]),
            "image_auroc_plain": float(roc_auc_score(labels, d_star)),
            "image_auroc_eq7": float(roc_auc_score(labels, scores)),
            "pixel_auroc": float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))),
            "w_median": float(np.median(ws)),
            "projection_dim": int(proj_dim),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--categories", default="leather,screw")
    ap.add_argument("--backbones", default="wide_resnet50_2,resnet50")
    ap.add_argument("--coreset-pcts", default="0.1")
    ap.add_argument("--out", default="../artifacts/curve.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None,
                    help="one grid point as category:backbone:n_shot:seed, for the smoke test")
    args = ap.parse_args()

    repro.tqdm = quiet_tqdm
    device = torch.device(args.device)
    pcts = [float(x) for x in args.coreset_pcts.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curve = load_curve(out_path)
    done = {point_key(r) for r in curve}
    print(f"device={device} pcts={pcts} resuming with {len(curve)} records already written",
          flush=True)

    only = args.only.split(":") if args.only else None
    t_start = time.time()

    for backbone in args.backbones.split(","):
        if only and only[1] != backbone:
            continue
        sha = repro.weights_sha256(backbone)
        model, taps = repro.build_extractor(device, backbone)
        assert not model.training, "backbone must be in eval mode"
        print(f"\n=== {backbone} weights_sha256={sha[:12]} ===", flush=True)

        for category in args.categories.split(","):
            if only and only[0] != category:
                continue
            train_paths, test_items = repro.list_dataset(args.data_root, category)
            labels = np.array([it[1] for it in test_items])
            masks = np.stack([repro.load_mask(it[2]) for it in test_items])
            print(f"{category}: {len(train_paths)} train, {len(test_items)} test "
                  f"({int((labels == 0).sum())} good + {int((labels == 1).sum())} defect)",
                  flush=True)

            for n_shot, seed in grid_for(train_paths):
                if only and (int(only[2]) != n_shot or int(only[3]) != seed):
                    continue
                todo = [p for p in pcts
                        if (category, backbone, n_shot, seed, round(p, 6)) not in done]
                if not todo:
                    print(f"  skip {category}/{backbone} n={n_shot} seed={seed}", flush=True)
                    continue
                t0 = time.time()
                recs = run_point(category, backbone, n_shot, seed, todo, model, taps,
                                 train_paths, test_items, labels, masks, device)
                for r in recs:
                    curve.append(r)
                    done.add(point_key(r))
                    print(f"  {category}/{backbone} n={r['n_shot']:>3} seed={r['seed']} "
                          f"bank={r['bank_rows']:>5}  plain={r['image_auroc_plain']:.4f}  "
                          f"eq7={r['image_auroc_eq7']:.4f}  pixel={r['pixel_auroc']:.4f}  "
                          f"w_med={r['w_median']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
                # write after every point, a crash keeps the work
                out_path.write_text(json.dumps(curve, indent=2))

    print(f"\nwrote {out_path} with {len(curve)} records in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
