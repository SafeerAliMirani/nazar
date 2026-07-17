"""Does layer3 dominate our concatenated feature?

The official PatchCore aggregation (common.py: MeanMapper then Aggregator)
pools each layer to 1024 and then 2048 -> 1024, so layer2 and layer3 end up
with an equal half of the feature. Our raw concat gives layer2 512 dims and
layer3 1024, so layer3 already owns two thirds of the dimensions before you
even look at the norms.

That matters because thread_side and manipulated_front defects are fine scale,
which is layer2 information, and those are exactly the ones we miss.

    python block_norms.py --data-root D:\\data\\mvtec_ad --category screw
"""

import argparse

import numpy as np
import torch

from repro import build_extractor, list_dataset, load_batch, merge_taps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", default="screw")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    train_paths, _ = list_dataset(args.data_root, args.category)
    model, taps = build_extractor(device, args.backbone)

    chunks = []
    for i in range(0, min(args.n, len(train_paths)), 16):
        model(load_batch(train_paths[i:i + 16]).to(device))
        f, _ = merge_taps(taps["layer2"], taps["layer3"])
        chunks.append(f.reshape(-1, f.shape[-1]).cpu())
    feats = torch.cat(chunks).numpy()

    l2, l3 = feats[:, :512], feats[:, 512:]
    n2 = np.linalg.norm(l2, axis=1)
    n3 = np.linalg.norm(l3, axis=1)
    tot = np.linalg.norm(feats, axis=1)

    print(f"{args.category} / {args.backbone}, {feats.shape[0]} patches x {feats.shape[1]} dims\n")
    print(f"layer2  dims 512 ({512/feats.shape[1]:.1%} of the vector)  mean L2 norm {n2.mean():.3f}")
    print(f"layer3  dims 1024 ({1024/feats.shape[1]:.1%} of the vector)  mean L2 norm {n3.mean():.3f}")
    print()
    # squared norm is what actually splits a euclidean distance
    share2 = (n2 ** 2).mean() / (tot ** 2).mean()
    share3 = (n3 ** 2).mean() / (tot ** 2).mean()
    print(f"share of squared norm, i.e. of every distance we compute:")
    print(f"  layer2 {share2:.1%}")
    print(f"  layer3 {share3:.1%}   <- official aggregation would make this 50%")
    print()
    print(f"per-dim mean squared value: layer2 {(l2**2).mean():.4f}  layer3 {(l3**2).mean():.4f}")
    print(f"layer3 : layer2 dominance ratio {share3/max(share2,1e-9):.2f}x")
    if share3 > 0.6:
        print("\nlayer3 dominates the distance metric. Fine scale layer2 detail, which is\n"
              "what thread_side and manipulated_front defects look like, is being diluted.")


if __name__ == "__main__":
    main()
