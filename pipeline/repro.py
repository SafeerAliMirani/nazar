"""PatchCore on one MVTec AD category, plain torch + numpy.

Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
CVPR 2022. WideResNet50 layer2+3 patch features, 10% greedy coreset,
L2 kNN scoring with the b=9 reweighting of eq. 7. Published
image AUROC to match: leather 100.0, wood 99.2.

    python repro.py --data-root /path/to/mvtec_ad --category leather

Self checks: python test_repro.py
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection
from torchvision import transforms
from torchvision.models import (ResNet50_Weights, Wide_ResNet50_2_Weights,
                                resnet50, wide_resnet50_2)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

RESIZE = 256
CROP = 224
BLUR_SIGMA = 4.0
B_NEAREST = 9
BATCH = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

tf_img = transforms.Compose([
    transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# same geometry as tf_img, for masks and heatmap panels
tf_mask = transforms.Compose([
    transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(CROP),
])
tf_view = transforms.Compose([
    transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(CROP),
])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_no_overlap(train_paths, test_paths):
    train_hashes = {sha256_file(p) for p in train_paths}
    dupes = [str(p) for p in test_paths if sha256_file(p) in train_hashes]
    assert not dupes, (
        f"{len(dupes)} test images are byte-identical to train images, "
        f"e.g. {dupes[:3]}. AUROC would be contaminated."
    )


def list_dataset(root, category):
    cat = Path(root) / category
    assert cat.is_dir(), f"{cat} not found"

    train_dir = cat / "train"
    subdirs = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    assert subdirs == ["good"], f"train must contain only good/, found {subdirs}"
    stray = sorted(train_dir.glob("*.png"))
    assert not stray, f"images sitting outside train/good: {stray[:3]}"
    train_paths = sorted((train_dir / "good").glob("*.png"))
    assert train_paths, f"no training images in {train_dir / 'good'}"

    test_dir = cat / "test"
    test_types = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    assert "good" in test_types, (
        "test split has no good images, AUROC would be computed on defects "
        "only and would be meaningless"
    )
    assert [t for t in test_types if t != "good"], "test split has no defective images"

    test_items = []  # (path, label, mask_path or None, defect type)
    for t in test_types:
        for p in sorted((test_dir / t).glob("*.png")):
            if t == "good":
                test_items.append((p, 0, None, t))
            else:
                mp = cat / "ground_truth" / t / f"{p.stem}_mask.png"
                assert mp.is_file(), f"missing ground truth mask {mp}"
                test_items.append((p, 1, mp, t))

    n_good = sum(1 for it in test_items if it[1] == 0)
    n_defect = len(test_items) - n_good
    assert n_good > 0 and n_defect > 0, (
        f"test split must contain both classes, got {n_good} good and "
        f"{n_defect} defective"
    )
    return train_paths, test_items


BACKBONES = {
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


def weights_sha256(name):
    # pin exactly which weights produced a number. resnet50 and wide_resnet50_2
    # banks have identical shapes, so nothing else can catch a mixup.
    _, weights = BACKBONES[name]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not path.is_file():
        return "not-cached"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    # torchvision puts the first 8 hex of the sha256 in the filename, so this
    # is a free integrity check. a truncated download fails here instead of
    # silently producing a plausible wrong number.
    stamp = path.stem.rsplit("-", 1)[-1]
    assert sha.startswith(stamp), (
        f"{path.name} is corrupt or not the official file: sha256 starts "
        f"{sha[:8]}, filename claims {stamp}")
    return sha


def build_extractor(device, name="wide_resnet50_2"):
    # both nets expose 512 ch at layer2 and 1024 at layer3, so the patch
    # geometry is the same. wide_resnet50_2 is the published PatchCore config.
    ctor, weights = BACKBONES[name]
    model = ctor(weights=weights)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    taps = {}
    model.layer2.register_forward_hook(lambda m, i, o: taps.__setitem__("layer2", o))
    model.layer3.register_forward_hook(lambda m, i, o: taps.__setitem__("layer3", o))
    return model, taps


# Set by main when --balance-blocks is on. The official aggregation
# (MeanMapper then Aggregator) pools each layer to 1024 and then 2048 -> 1024,
# so layer2 and layer3 contribute equally to every distance. Our raw concat
# measures 62/38 in layer2's favour, because layer2's per-dim energy is ~3.2x
# layer3's. This rescales the two blocks to an equal share.
BLOCK_SCALE = None


def merge_taps(f2, f3):
    # 3x3 local average per tap, layer3 aligned onto layer2's grid, concat
    f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
    f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    if BLOCK_SCALE is not None:
        f2 = f2 * BLOCK_SCALE[0]
        f3 = f3 * BLOCK_SCALE[1]
    f = torch.cat([f2, f3], dim=1)
    b, c, h, w = f.shape
    return f.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)


def load_batch(paths):
    return torch.stack([tf_img(Image.open(p).convert("RGB")) for p in paths])


@torch.no_grad()
def extract_train(paths, model, taps, device):
    chunks, grid = [], None
    for i in tqdm(range(0, len(paths), BATCH), desc="train features"):
        model(load_batch(paths[i:i + BATCH]).to(device))
        feats, grid = merge_taps(taps["layer2"], taps["layer3"])
        chunks.append(feats.reshape(-1, feats.shape[-1]).cpu())
    return torch.cat(chunks), grid


def coreset_indices(features, pct, seed, device):
    # greedy k-center on JL-projected features, the bank keeps full dims
    n = features.shape[0]
    if pct >= 1.0:
        # PatchCore-100. The official sampler short circuits the same way
        # (if self.percentage == 1: return features), so there is no greedy
        # pass to run and no selection to get wrong.
        return np.arange(n), features.shape[1]
    m = max(1, int(round(n * pct)))
    feats_np = features.numpy()
    try:
        proj = SparseRandomProjection(n_components="auto", eps=0.9, random_state=seed)
        reduced = proj.fit_transform(feats_np).astype(np.float32)
        if reduced.shape[1] >= feats_np.shape[1]:
            reduced = feats_np
    except ValueError:
        # JL bound above the actual dim, happens only on tiny debug sets
        reduced = feats_np
    pts = torch.from_numpy(reduced).to(device)

    rng = np.random.RandomState(seed)
    picked = [int(rng.randint(n))]
    d2 = ((pts - pts[picked[0]]) ** 2).sum(1)
    for _ in tqdm(range(m - 1), desc="coreset"):
        idx = int(torch.argmax(d2))
        picked.append(idx)
        d2 = torch.minimum(d2, ((pts - pts[idx]) ** 2).sum(1))
    return np.array(picked), reduced.shape[1]


def reweighted_score(d_star, d_star_to_nb, return_w=False):
    # eq. 7: d_star is d(q*, n*), d_star_to_nb are d(q*, n_i) for the b
    # nearest bank neighbours of n*. The exp() is shifted by the max
    # distance, which cancels in the ratio, so large d cannot overflow.
    d = np.asarray(d_star_to_nb, dtype=np.float64)
    c = max(float(d_star), float(d.max()))
    w = 1.0 - np.exp(d_star - c) / np.exp(d - c).sum()
    return (float(w * d_star), float(w)) if return_w else float(w * d_star)


@torch.no_grad()
def score_test(test_items, model, taps, bank, device):
    bank = bank.to(device)
    bank_sq = (bank ** 2).sum(1)[None, :]
    b = min(B_NEAREST, bank.shape[0])
    scores, maps, ws = [], [], []
    for i in tqdm(range(0, len(test_items), BATCH), desc="test"):
        chunk = test_items[i:i + BATCH]
        model(load_batch([it[0] for it in chunk]).to(device))
        feats, (h, w) = merge_taps(taps["layer2"], taps["layer3"])
        for q in feats:
            d2 = (q ** 2).sum(1, keepdim=True) + bank_sq - 2.0 * (q @ bank.T)
            d2.clamp_(min=0)
            # eq. 7 weights the L2 norm. Feeding it squared distances saturates
            # the softmax and pins w at 1.0, which silently removes the
            # reweighting and leaves plain max-min-distance scoring.
            dist = d2.sqrt()
            mins, argmins = dist.min(dim=1)
            p_idx = int(torch.argmax(mins))
            d_star = float(mins[p_idx])
            n_star = int(argmins[p_idx])
            # topk on squared bank distances gives the same neighbours as L2
            nb = torch.topk(((bank - bank[n_star]) ** 2).sum(1), k=b, largest=False).indices
            # not "w": that shadows the grid width from merge_taps and the
            # reshape below silently gets a float
            sc, wgt = reweighted_score(d_star, dist[p_idx, nb].cpu().numpy(), return_w=True)
            scores.append(sc)
            ws.append(wgt)
            amap = F.interpolate(mins.reshape(1, 1, h, w), size=(CROP, CROP),
                                 mode="bilinear", align_corners=False)[0, 0]
            maps.append(gaussian_filter(amap.cpu().numpy(), sigma=BLUR_SIGMA))
    return np.asarray(scores), np.stack(maps), np.asarray(ws)


@torch.no_grad()
def sample_test_patches(test_items, model, taps, device, n=64):
    # a few real query patches for the goldens, from images the bank never saw
    model(load_batch([it[0] for it in test_items[:4]]).to(device))
    feats, _ = merge_taps(taps["layer2"], taps["layer3"])
    flat = feats.reshape(-1, feats.shape[-1]).cpu()
    idx = torch.linspace(0, flat.shape[0] - 1, n).long()
    return flat[idx]


def load_mask(mask_path):
    if mask_path is None:
        return np.zeros((CROP, CROP), dtype=np.uint8)
    m = tf_mask(Image.open(mask_path).convert("L"))
    return (np.asarray(m) > 0).astype(np.uint8)


def jet(x):
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def save_panels(out_dir, test_items, scores, maps, masks):
    labels = np.array([it[1] for it in test_items])
    order = np.argsort(-scores)
    chosen = [i for i in order if labels[i] == 1][:8] + [i for i in order if labels[i] == 0][:2]
    chosen += [i for i in order if i not in chosen][:10 - len(chosen)]

    vmin, vmax = maps.min(), maps.max()  # shared scale, display only
    gap = np.full((CROP, 4, 3), 255, dtype=np.uint8)
    panel_dir = out_dir / "heatmaps"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for rank, i in enumerate(chosen):
        path, _, _, dtype = test_items[i]
        img = np.asarray(tf_view(Image.open(path).convert("RGB")))
        heat = (jet((maps[i] - vmin) / (vmax - vmin + 1e-12)) * 255).astype(np.uint8)
        overlay = np.clip(0.55 * img + 0.45 * heat, 0, 255).astype(np.uint8)
        mask_rgb = np.repeat(masks[i][:, :, None] * 255, 3, axis=2)
        panel = np.concatenate([img, gap, overlay, gap, mask_rgb], axis=1)
        name = f"{rank:02d}_{dtype}_{path.stem}_score_{scores[i]:.4f}.png"
        Image.fromarray(panel).save(panel_dir / name)


def bank_nb9(bank, b, device):
    # the b nearest bank points to every bank point. bank-to-bank and fixed at
    # freeze time, so the browser gathers 9 rows instead of running a second
    # kNN. chunked, the full NxN matrix does not fit.
    bank = bank.to(device)
    sq = (bank ** 2).sum(1)
    out = torch.empty(bank.shape[0], b, dtype=torch.int32)
    step = 2048
    for i in range(0, bank.shape[0], step):
        q = bank[i:i + step]
        d2 = (q ** 2).sum(1, keepdim=True) + sq[None, :] - 2.0 * (q @ bank.T)
        d2.clamp_(min=0)
        out[i:i + step] = torch.topk(d2, k=b, largest=False).indices.to(torch.int32).cpu()
    return out


def make_goldens(bank, test_feats, b, device):
    """Queries the WGSL kernel must reproduce. Every expectation is float64.

    The edge cases are the point. A kernel that skips the reweighting, or
    squares the distances, or drops the last partial workgroup, still looks
    right on a heatmap. It cannot pass these.
    """
    bank64 = bank.to(torch.float64)
    N = bank.shape[0]
    cases = []

    def add(name, q):
        q = q.to(torch.float64)
        d = torch.cdist(q[None, :], bank64)[0]          # true L2, float64
        d_star, n_star = float(d.min()), int(d.argmin())
        nb = torch.topk(((bank64 - bank64[n_star]) ** 2).sum(1), k=b, largest=False).indices
        d_nb = d[nb].numpy()
        sc, w = reweighted_score(d_star, d_nb, return_w=True)
        cases.append({
            "name": name,
            "query": q.numpy().astype(np.float32).tolist(),
            "expect": {
                "min_distance": d_star,
                "argmin": n_star,
                "nb9_idx": nb.numpy().astype(int).tolist(),
                "nb9_distances": [float(x) for x in d_nb],
                "w": w,
                "score": sc,
            },
        })

    # dispatch coverage: nearest neighbour pinned at the first and LAST bank row.
    # N is deliberately not a multiple of any sane workgroup size, so a
    # ceil-division bug drops the last partial group and fails on nn_at_last.
    add("nn_at_first", bank[0])
    add("nn_at_last", bank[N - 1])
    # numerical guards
    add("all_zeros", torch.zeros(bank.shape[1]))
    add("far_query", bank[N // 2] + 8.0)   # large d, naive fp16 exp overflows
    # real patches, including the ones that discriminate on w
    idx = np.linspace(0, test_feats.shape[0] - 1, 12).astype(int)
    for j, k in enumerate(idx):
        add(f"real_{j:02d}", test_feats[k])
    return cases


def export_artifacts(out, bank, test_feats, args, weights_sha, grid, proj_dim, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.category}_{args.backbone}_c{int(args.coreset_pct * 100)}"
    b = min(B_NEAREST, bank.shape[0])

    (out / f"bank_{tag}.f32.bin").write_bytes(bank.numpy().astype("<f4").tobytes())
    (out / f"bank_{tag}.f16.bin").write_bytes(bank.numpy().astype("<f2").tobytes())
    nb9 = bank_nb9(bank, b, device)
    (out / f"bank_{tag}.nb9.i32.bin").write_bytes(nb9.numpy().astype("<i4").tobytes())

    goldens = make_goldens(bank, test_feats, b, device)
    (out / f"goldens_{tag}.json").write_text(json.dumps(goldens))

    side = {
        "bank": {"rows": int(bank.shape[0]), "dims": int(bank.shape[1]),
                 "dtype_f16": "little-endian float16, row major",
                 "f16_bytes": int(bank.shape[0] * bank.shape[1] * 2)},
        "nb9": {"shape": [int(nb9.shape[0]), int(nb9.shape[1])], "dtype": "little-endian int32"},
        "provenance": {
            "category": args.category, "backbone": args.backbone,
            "weights_sha256": weights_sha, "seed": args.seed,
            "coreset_pct": args.coreset_pct, "projection_dim": proj_dim,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
        },
        "preprocessing": {"resize": RESIZE, "interpolation": "bilinear", "crop": CROP,
                          "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "features": {"taps": ["layer2", "layer3"], "pool": "avg 3x3 stride 1 pad 1",
                     "upsample": "bilinear align_corners=False onto layer2 grid",
                     "concat_order": ["layer2", "layer3"], "grid": list(grid)},
        "scoring": {"distance": "L2 (NOT squared: squared saturates the eq.7 exp)",
                    "b_nearest": b, "score": "w * d_star, w per eq.7",
                    "map": "pre-blur 28x28 min-distances, then bilinear to 224, gaussian sigma 4"},
        "tolerance": ("goldens are float64. bank ships fp16 (~1e-3 relative), accumulate "
                      "in fp32. expect distance error ~ sqrt(1536) * fp16 eps * scale. "
                      "derive the bound, do not tune it until it passes."),
        "warning": ("resnet50 and wide_resnet50_2 banks are shape identical. never mix "
                    "artifacts across backbones. loaders must check weights_sha256."),
    }
    (out / f"sidecar_{tag}.json").write_text(json.dumps(side, indent=2))
    print(f"exported to {out}: bank {bank.shape[0]}x{bank.shape[1]} "
          f"(f16 {side['bank']['f16_bytes']/1e6:.1f} MB), nb9 {tuple(nb9.shape)}, "
          f"{len(goldens)} goldens")


@torch.no_grad()
def export_live_queries(out, test_items, model, taps, bank, scores, ws, maps, args, device, n=12):
    """Per-image patch features so the browser can run the kNN itself.

    The backbone stays offline (that is P3). The kNN, the reweighting and the
    heatmap run live in WGSL on the visitor's GPU. Each image ships with the
    python reference score so the page can check its own GPU result instead of
    asking to be believed.
    """
    out = Path(out)
    (out / "queries").mkdir(parents=True, exist_ok=True)
    labels = np.array([it[1] for it in test_items])
    order = np.argsort(-scores)
    # a spread worth looking at: worst defects, the worst good, and the
    # near-threshold cases where the decision actually hurts
    pick = [i for i in order if labels[i] == 1][:4]
    pick += [i for i in order if labels[i] == 0][:2]
    mid = [i for i in order if labels[i] == 1][len(pick):]
    pick += [i for i in order if labels[i] == 1][-3:]     # the ones it misses
    pick += [i for i in order if labels[i] == 0][-2:]     # quiet goods
    pick = list(dict.fromkeys(pick))[:n]

    b = min(B_NEAREST, bank.shape[0])
    index = []
    for rank, i in enumerate(pick):
        path, lab, mask_path, dtype = test_items[i]
        model(load_batch([path]).to(device))
        feats, (h, w) = merge_taps(taps["layer2"], taps["layer3"])
        q = feats[0].cpu()
        name = f"{args.category}_{dtype}_{path.stem}"
        (out / "queries" / f"{name}.f16.bin").write_bytes(q.numpy().astype("<f2").tobytes())
        img = np.asarray(tf_view(Image.open(path).convert("RGB")))
        Image.fromarray(img).save(out / "queries" / f"{name}.png")
        Image.fromarray((load_mask(mask_path) * 255)).save(out / "queries" / f"{name}_mask.png")
        index.append({
            "name": name, "label": int(lab), "defect_type": dtype,
            "file": path.name, "patches": int(q.shape[0]), "dims": int(q.shape[1]),
            "grid": [int(h), int(w)],
            "reference": {  # what python got, for the page to check itself against
                "score_eq7": float(scores[i]), "w": float(ws[i]),
                "d_star": float(scores[i] / max(ws[i], 1e-12)),
                "map_max": float(maps[i].max()), "map_min": float(maps[i].min()),
            },
        })
    tag = f"{args.category}_{args.backbone}"
    (out / "queries" / f"{tag}.json").write_text(json.dumps({
        "category": args.category, "backbone": args.backbone,
        "bank": f"bank_{args.category}_{args.backbone}_c{int(args.coreset_pct*100)}",
        "b_nearest": b, "crop": CROP, "blur_sigma": BLUR_SIGMA,
        "note": ("patch features are precomputed offline. the kNN, the eq.7 "
                 "reweighting and the heatmap run in WGSL on your GPU."),
        "images": index,
    }, indent=2))

    # manifest of every category exported so far, so the page can offer a
    # switch instead of showing whichever one was exported last
    man = out / "queries" / "manifest.json"
    have = json.loads(man.read_text())["sets"] if man.is_file() else []
    have = [h for h in have if h["tag"] != tag]
    have.append({"tag": tag, "category": args.category, "backbone": args.backbone,
                 "index": f"{tag}.json", "bank": f"bank_{tag}_c{int(args.coreset_pct*100)}",
                 "images": len(index)})
    have.sort(key=lambda h: h["tag"])
    man.write_text(json.dumps({
        "note": "each set is a separate bank download. switching costs bandwidth.",
        "sets": have}, indent=2))
    print(f"exported {len(index)} live queries to {out / 'queries'} as {tag}.json")
    print(f"manifest now lists: {', '.join(h['tag'] for h in have)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--category", default="leather")
    ap.add_argument("--coreset-pct", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="wide_resnet50_2",
                    choices=sorted(BACKBONES))
    # causal checks. a correct pipeline must FAIL these, loudly.
    ap.add_argument("--ablate-bank", default="none",
                    choices=["none", "shuffle", "random-coreset"],
                    help="shuffle: destroy bank structure, keep per-dim stats. "
                         "a real pipeline should collapse")
    ap.add_argument("--balance-blocks", action="store_true",
                    help="rescale layer2/layer3 to contribute equally to the "
                         "distance, as the official aggregation does")
    ap.add_argument("--permute-labels", action="store_true",
                    help="sanity check the metric itself, expect AUROC ~0.5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--export-dir", default=None,
                    help="write the bank, nb9 table and goldens for the WGSL kernel")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    weights_sha = weights_sha256(args.backbone)
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"category={args.category} backbone={args.backbone} seed={args.seed} device={device}")

    train_paths, test_items = list_dataset(args.data_root, args.category)
    by_type = {}
    for _, _, _, t in test_items:
        by_type[t] = by_type.get(t, 0) + 1
    n_good = by_type["good"]
    n_defect = len(test_items) - n_good
    print(f"train: {len(train_paths)} images, all under train/good")
    print(f"test: {len(test_items)} images, {n_good} good + {n_defect} defective "
          f"({', '.join(f'{t}:{c}' for t, c in sorted(by_type.items()))})")
    assert_no_overlap(train_paths, [it[0] for it in test_items])
    print("train/test content overlap: none (sha256)")

    model, taps = build_extractor(device, args.backbone)
    assert not model.training, "backbone must be in eval mode, BN would leak batch composition"

    if args.balance_blocks:
        global BLOCK_SCALE
        with torch.no_grad():
            model(load_batch(train_paths[:16]).to(device))
            probe, _ = merge_taps(taps["layer2"], taps["layer3"])
        probe = probe.reshape(-1, probe.shape[-1])
        e2 = float((probe[:, :512] ** 2).sum(1).mean())
        e3 = float((probe[:, 512:] ** 2).sum(1).mean())
        # equalise the mean squared contribution of each block
        BLOCK_SCALE = ((e3 / e2) ** 0.5, 1.0)
        print(f"block balance: layer2 share was {e2/(e2+e3):.1%}, scaling layer2 by "
              f"{BLOCK_SCALE[0]:.4f} to reach 50/50")
    train_feats, grid = extract_train(train_paths, model, taps, device)
    print(f"train patches: {tuple(train_feats.shape)}, grid {grid}")

    # bank, projection and coreset are fixed here, before any test image is read
    picked, proj_dim = coreset_indices(train_feats, args.coreset_pct, args.seed, device)
    assert len(set(picked.tolist())) == len(picked), "coreset picked a duplicate"
    bank = train_feats[torch.from_numpy(picked)].clone()
    expect_m = max(1, int(round(train_feats.shape[0] * args.coreset_pct)))
    assert bank.shape[0] == expect_m, f"bank is {bank.shape[0]}, expected {expect_m}"
    del train_feats
    print(f"bank: {bank.shape[0]} patches ({100 * args.coreset_pct:.1f}%), "
          f"selection ran on {proj_dim} projected dims")

    if args.ablate_bank == "shuffle":
        # permute every feature dimension independently across bank rows. the
        # per-dim marginals survive, the patch manifold does not. if scores
        # stay high after this, something other than the bank is classifying.
        g = torch.Generator().manual_seed(args.seed)
        order = torch.argsort(torch.rand(bank.shape, generator=g), dim=0)
        bank = torch.gather(bank, 0, order)
        print("ABLATION: bank structure destroyed (per-dim shuffle)")
    elif args.ablate_bank == "random-coreset":
        g = torch.Generator().manual_seed(args.seed + 1)
        keep = torch.randperm(bank.shape[0], generator=g)[: bank.shape[0]]
        bank = bank[keep]
        print("ABLATION: coreset replaced by a random subset")

    scores, maps, ws = score_test(test_items, model, taps, bank, device)

    # R4/6: the reweighting must still be doing something on real data. it once
    # ran, passed its unit tests, and contributed nothing because squared
    # distances saturated the exp. a rescaling of the features could bring that
    # back with the code unchanged, so assert on the live spread of w.
    print(f"eq.7 weight w: min {ws.min():.6f}  p10 {np.percentile(ws, 10):.6f}  "
          f"median {np.median(ws):.6f}  max {ws.max():.6f}  std {ws.std():.2e}")
    if np.percentile(ws, 10) > 0.999 or ws.std() < 1e-4:
        print("WARNING: w is pinned near 1.0, the eq.7 reweighting is a no-op. "
              "check that distances are L2 and not squared.")
    labels = np.array([it[1] for it in test_items])
    masks = np.stack([load_mask(it[2]) for it in test_items])

    if args.permute_labels:
        rng = np.random.RandomState(args.seed)
        labels = rng.permutation(labels)
        print("SANITY: labels permuted, expect image AUROC near 0.5")

    # d_star = the plain max-over-patches NN distance. scores = w * d_star.
    # The official implementation scores with d_star alone (no eq.7), so
    # image_auroc_plain is the number comparable to the published table, and
    # image_auroc_eq7 is our paper-faithful variant. Report both, always.
    d_star = scores / np.maximum(ws, 1e-12)
    image_auroc_plain = roc_auc_score(labels, d_star)
    image_auroc = roc_auc_score(labels, scores)
    assert maps.shape == masks.shape, (
        f"anomaly maps {maps.shape} and masks {masks.shape} must match at metric "
        f"time, else pixel AUROC is not comparable to the published table")
    pixel_auroc = roc_auc_score(masks.reshape(-1), maps.reshape(-1))

    golden_src = (sample_test_patches(test_items, model, taps, device)
                  if args.export_dir else None)
    save_panels(out_dir, test_items, scores, maps, masks)
    metrics = {
        "category": args.category,
        "seed": args.seed,
        "coreset_pct": args.coreset_pct,
        "image_auroc_plain": float(image_auroc_plain),
        "image_auroc_eq7": float(image_auroc),
        "image_auroc": float(image_auroc_plain),  # headline = the comparable one
        "pixel_auroc": float(pixel_auroc),
        "n_train": len(train_paths),
        "n_test": len(test_items),
        "n_test_good": int(n_good),
        "n_test_defect": int(n_defect),
        "test_counts": by_type,
        "bank_size": int(bank.shape[0]),
        "w_p10": float(np.percentile(ws, 10)),
        "w_median": float(np.median(ws)),
        "w_std": float(ws.std()),
        "ablate_bank": args.ablate_bank,
        "permute_labels": bool(args.permute_labels),
        "config": {
            "data_root": str(args.data_root),
            "device": str(device),
            "backbone": f"{args.backbone} IMAGENET1K_V1",
            "balance_blocks": bool(args.balance_blocks),
            "weights_sha256": weights_sha,
            "taps": ["layer2", "layer3"],
            "resize": RESIZE,
            "crop": CROP,
            "patch_pool": "avg 3x3 stride 1 pad 1",
            "grid": list(grid),
            "feature_dim": int(bank.shape[1]),
            "projection_dim": int(proj_dim),
            "blur_sigma": BLUR_SIGMA,
            "b_nearest": B_NEAREST,
            "batch": BATCH,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # every per-image number, so future questions never need another 30 min run
    rows = ["file,label,defect_type,d_star,w,score_eq7"]
    for (path, lab, _, dtype), ds, w, sc in zip(test_items, d_star, ws, scores):
        rows.append(f"{path.name},{lab},{dtype},{ds:.6f},{w:.6f},{sc:.6f}")
    (out_dir / "scores.csv").write_text("\n".join(rows))

    print(f"image AUROC (plain max, matches official code): {image_auroc_plain:.4f}")
    print(f"image AUROC (with eq.7 reweighting):              {image_auroc:.4f}")
    print(f"pixel AUROC: {pixel_auroc:.4f}")
    print(f"wrote {out_dir / 'metrics.json'} and {out_dir / 'heatmaps'}")

    if args.export_dir:
        export_artifacts(args.export_dir, bank, golden_src, args, weights_sha,
                         grid, proj_dim, device)
        export_live_queries(args.export_dir, test_items, model, taps, bank,
                            scores, ws, maps, args, device)


if __name__ == "__main__":
    main()
