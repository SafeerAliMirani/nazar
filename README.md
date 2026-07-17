<div align="center">

# Nazar

**All fifteen kinds of real parts, an inspector that has never seen a defect, scored on your GPU.**

[![Live demo](https://img.shields.io/badge/live-nazar.pages.dev-c0392b?style=for-the-badge)](https://nazar.pages.dev)
&nbsp;
![WebGPU](https://img.shields.io/badge/WebGPU-raw%20WGSL-1f6feb?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/dependencies-none-2ea043?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-8957e5?style=for-the-badge)

</div>

Nazar learns what a good part looks like from photographs of good parts and nothing else, then scores a new part by how far its worst patch sits from that memory. Pick a part in the browser and a hand-written WGSL compute shader runs the nearest neighbour search over the memory bank on your own graphics card, about 19.7 million distances in roughly 150 ms, and paints the heatmap where the surface is least like anything it has seen. Every part ships with the score the offline pipeline computed for it, and the page checks its own GPU answer against that number in front of you.

## What you are looking at

- **Real data.** MVTec AD, the standard industrial inspection benchmark, all fifteen categories: screws, capsules, pills, cable, metal nuts, transistors, leather, hazelnuts, bottles, carpet, grid, tile, toothbrushes, wood and zippers, photographed on one bench under one fixed light. CC BY-NC-SA 4.0, Bergmann et al. Nothing here comes from a factory, and a production line looks nothing like that bench.
- **Trained on good parts only.** The memory bank for the screw was built from 320 defect-free photographs, and no defect ever went into it. This is the constraint a real line imposes: good parts pile up by the thousand and the next fault is one nobody has photographed.
- **The search runs on your GPU.** 784 patches per part, 1536 dimensions each, against a bank of 3,340 to 6,131 rows. A WGSL compute shader takes one workgroup per patch, strides the bank, and reduces to the nearest row. The eq.7 reweighting and the heatmap follow. The backbone that turns a photograph into those numbers does not run in the browser, its features are precomputed offline, and the page says so.
- **A bank five times smaller than the paper's.** The published default keeps 10 percent of patches, 14.5 to 94.2 MB per category and 874 MB across the fifteen. Nazar ships 2 percent, 2.9 to 18.8 MB and 175 MB in total, and the mean image AUROC does not drop: 98.874 against 98.805 at 10 percent. The categories spread over 4.64 points at 2 percent though, so a 0.07 point difference between means does not establish 2 percent as better in general.
- **Honest by design.** The reproduction is measured against the published table rather than asserted, the threshold panel shows the defects no setting can catch, and the few-shot curve is printed as it came out, including the part where it is worse than a coin toss.

## The numbers

Image AUROC on all fifteen MVTec AD categories, our 10 percent run against the paper's 10 percent, wide_resnet50_2 at 224 by 224. The paper publishes at 1, 10 and 25 percent, so the shipped 2 percent bank has no published counterpart and the gap column compares like with like. Published figures from Roth et al., CVPR 2022 supplementary Table S1.

| category | ours, 2% shipped | ours, 10% | published, 10% | gap |
|---|---|---|---|---|
| bottle | 100.00 | 100.00 | 100.0 | +0.00 |
| cable | 99.87 | 99.61 | 99.4 | +0.21 |
| capsule | 97.65 | 98.05 | 97.8 | +0.24 |
| carpet | 98.96 | 99.00 | 98.7 | +0.30 |
| grid | 97.58 | 97.58 | 97.9 | -0.32 |
| hazelnut | 100.00 | 100.00 | 100.0 | +0.00 |
| leather | 100.00 | 100.00 | 100.0 | +0.00 |
| metal_nut | 99.85 | 99.61 | 100.0 | -0.39 |
| pill | 95.36 | 94.90 | 96.0 | -1.10 |
| screw | 96.68 | 96.39 | 97.0 | -0.61 |
| tile | 98.38 | 98.85 | 98.9 | -0.05 |
| toothbrush | 100.00 | 99.72 | 99.7 | +0.02 |
| transistor | 100.00 | 99.83 | 100.0 | -0.17 |
| wood | 99.12 | 99.12 | 99.0 | +0.12 |
| zipper | 99.66 | 99.42 | 99.5 | -0.08 |
| **mean** | **98.87** | **98.80** | **98.93** | **-0.12** |

Fourteen of the fifteen land within 0.61 points of the published figure. Pill is the one that does not, at -1.10.

Two results worth reading twice. The first is the few-shot curve on the screw: one good part in the memory scores 0.482, five score 0.493, ten score 0.570, twenty-five score 0.706, and all 320 reach 0.964. Under five parts it is worse than a coin toss, because with a handful of reference screws the search measures pose and rotation rather than defects, and an oddly rotated good screw then looks stranger than a real one. The second is leather, which scores a perfect 1.0000 and still scores 0.9487 with its memory deliberately scrambled along every feature dimension. An easy surface flatters any method, so the screw is the one worth watching.

## How it works

1. `get_leather.py` pulls one MVTec category and lays it out the way the pipeline expects. Resumable, because the connection here is not.
2. `repro.py` extracts patch features from wide_resnet50_2 blocks 2 and 3, builds the memory bank by greedy k-center coreset, scores the test split with an exact kNN, and reports both the plain max score the official code uses and the paper's eq.7 reweighting. `--export-dir` writes the bank in float16, a precomputed 9 nearest neighbour table, float64 goldens for the kernel, and the per-part features the browser scores live.
3. `sweep.py` measures the few-shot curve and `report.py` the compression curve, both against the published tables.
4. `publish_to_app.py` is the only path from a run to the page. It refuses to publish anything that is not the shipped configuration, so a stray backbone cannot reach the site by accident.
5. `build_dist.py` assembles `dist`, the only folder that should ever be deployed. The page loads the bank, runs the kNN and the heatmap in raw WebGPU, and checks itself against the offline score. No three.js, no framework, no build step.

## Run it

```bash
python pipeline/get_leather.py --out D:\data\mvtec_ad --category screw
python pipeline/repro.py --data-root D:\data\mvtec_ad --category screw \
    --backbone wide_resnet50_2 --coreset-pct 0.02 --export-dir ../artifacts
python pipeline/test_repro.py       # self checks, including the eq.7 regression
python pipeline/build_dist.py       # then serve dist/
```

The kernel has its own parity harness at `web/harness`, which loads the goldens and prints the WGSL result against the float64 reference case by case, including the nearest neighbour pinned at the last bank row that a ceil-division bug would silently drop.

## Notes

The dataset is MVTec AD and it is CC BY-NC-SA 4.0. The banks and the heatmaps are derived from those images and carry the same licence, which is why they are not in this repository. The code is MIT. Download the archive from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) rather than a mirror.

The method is PatchCore, Roth et al., CVPR 2022. One thing worth knowing if you read the paper and then this code: the released implementation scores an image with a plain max over the patch distances and never applies eq.7, so the plain number is the one comparable to the published table, and both are reported here.

## Author

Dr Safeer Ali Mirani

[safeer.ali.mirani@gmail.com](mailto:safeer.ali.mirani@gmail.com) · [Portfolio](https://safeeralimirani.pages.dev) · [GitHub](https://github.com/SafeerAliMirani) · [LinkedIn](https://www.linkedin.com/in/safeeralimirani)
