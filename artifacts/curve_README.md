# curve.json

The few-shot curve: image AUROC as a function of how many good parts the
memory bank was built from. Produced by `pipeline/sweep.py`, which imports
every piece of feature extraction, coreset selection and scoring from
`pipeline/repro.py` so the numbers here and the numbers that passed the P0
repro gate come from the same code.

## Command

```
cd Nazar/pipeline
python sweep.py --data-root D:\data\mvtec_ad \
                --categories leather,screw \
                --backbones wide_resnet50_2,resnet50 \
                --out ../artifacts/curve.json
```

Resumable. Grid points already in `curve.json` are skipped, so an interrupted
sweep can be restarted with the same command.

## Columns

Flat list of records, one per grid point.

| field | meaning |
|---|---|
| `category` | MVTec AD category, `leather` or `screw` |
| `backbone` | `wide_resnet50_2` (the published PatchCore config) or `resnet50` |
| `n_shot` | number of good training images the bank was built from. The largest value per category is the full train split (leather 245, screw 320), not a sampled subset |
| `seed` | seed for the image draw, the JL projection and the greedy init |
| `coreset_pct` | coreset fraction, 0.1 throughout (PatchCore-10) |
| `bank_rows` | rows in the memory bank. Asserted to equal `round(n_shot * 784 * coreset_pct)` exactly at every grid point |
| **`image_auroc_plain`** | **the headline.** Image AUROC from the plain max over per-patch NN distance |
| `image_auroc_eq7` | image AUROC using the paper's eq.7 reweighting |
| `pixel_auroc` | pixel AUROC of the blurred anomaly map against the ground truth mask |
| `w_median` | median eq.7 weight on real data. A no-op guard: if this pins at 1.0 the reweighting has silently stopped contributing |
| `projection_dim` | dims the greedy selection ran in after the JL projection. The bank itself keeps all 1536 |

### Why `image_auroc_plain` is the headline

The official PatchCore implementation (`amazon-science/patchcore-inspection`)
does **not** implement eq.7. Its image score is a plain max over the raw
per-patch NN distance. `image_auroc_plain` is therefore the number that is
comparable to the published table; `image_auroc_eq7` is our paper-faithful
variant, reported alongside so the difference stays visible. Any comparison to
a published number must use `image_auroc_plain`.

## Seed protocol

Seeds exist to randomise **which** training images are drawn, so they only
carry information where there is a draw to randomise.

- **n_shot in {1, 2, 5, 10, 25}: 3 seeds** (0, 1, 2).
- **n_shot = full: 1 seed.** The whole train split is used, so there is no draw
  left; the selection is deterministic given a fixed init.

Within a seed the subsets are **nested**: the seed permutes the train split
once and `n_shot` takes the first n. So the 1-shot image is also in that seed's
2-shot set. This keeps the climb attributable to n rather than to which images
happened to be drawn. Each n-subset is still a uniform random n-subset of the
train split; nesting only correlates the estimates along the curve.

**Report the band as min/max across the 3 seeds.** Three samples do not support
a standard error, and quoting one would be theatre.

## The coreset costs nothing on the k axis

Greedy k-center is prefix-nested: the first m selections at budget K are
identical to the selections at budget m. `sweep.py` runs one greedy selection
per (category, backbone, n_shot, seed) at the largest budget needed and slices
prefixes for anything smaller. If a future sweep adds coreset percentages, they
are free.

## Comparison to the published few-shot curve: read this before plotting

PatchCore supplementary **Table S5** (PatchCore-10, image AUROC) reports:

| n_shot | 1 | 2 | 5 | 10 | 16 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| image AUROC | 83.4 +/- 0.6 | 86.4 +/- 0.9 | 90.8 +/- 0.8 | 93.6 +/- 0.6 | 95.4 | 95.8 | 97.5 |

**Those are means over all 15 MVTec categories. This file contains
per-category numbers. They are not a like-for-like comparison and must never
be presented as if they should match.**

A per-category curve can sit far above the S5 mean (leather is at 1.0 from one
shot) or below it, without either being evidence of anything. Reproducing S5
would require running all 15 categories and averaging. We ran 2.

## The measured curve

`image_auroc_plain`, mean over seeds with the min/max band, 64 records total.

| n_shot | bank rows | leather / WRN-50-2 | screw / WRN-50-2 | screw / resnet50 |
|---|---|---|---|---|
| 1 | 78 | 1.0000 | 0.4817 (0.442 to 0.556) | 0.4578 (0.425 to 0.522) |
| 2 | 157 | 1.0000 | 0.4644 (0.431 to 0.522) | 0.4613 (0.434 to 0.513) |
| 5 | 392 | 1.0000 | 0.4929 (0.448 to 0.580) | 0.4947 (0.453 to 0.557) |
| 10 | 784 | 1.0000 | 0.5695 (0.494 to 0.688) | 0.5584 (0.503 to 0.655) |
| 25 | 1960 | 1.0000 | 0.7060 (0.655 to 0.771) | 0.6868 (0.626 to 0.752) |
| full (245 / 320) | 19208 / 25088 | 1.0000 | 0.9639 | 0.9498 |

leather / resnet50 is 1.0000 everywhere except n=1, which averages 0.9997.

Two things to read off it:

- **Leather saturates.** It is at image AUROC 1.0 from a single training image
  and stays there. It is in the file as the easy contrast, and it is exactly
  why leather cannot carry the hero chart. A structurally destroyed memory bank
  still scores 0.94 on leather (REVIEW_V2 section 1), so a high leather number
  is close to uninformative.
- **Screw is the category that carries the result.** It starts *at or below
  chance* (0.46 to 0.48 mean for n = 1, 2, 5), stays there until about 10
  images, and only then climbs, reaching 0.71 at 25 and 0.96 at the full 320.
  Below-chance at small n is a real effect, not a bug: with a handful of
  reference screws the nearest-neighbour distance is dominated by pose and
  rotation mismatch rather than by defects, so an unusually posed good screw
  outscores a defective one. The seed band is wide (0.494 to 0.688 at n=10),
  which is itself part of the finding: at small n the result depends heavily on
  *which* parts you happened to photograph.

### Full-train points reproduce the P0 gate exactly

The n = full rows are not a separate experiment. They were cross-checked
against the standalone `repro.py` runs in `pipeline/runs/` and agree
bit-for-bit (difference 0.0, not merely within tolerance):

| run | repro.py | sweep.py |
|---|---|---|
| `runs/screw_wrn` | plain 0.963927 | plain 0.963927 |
| `runs/screw_r50` | 0.944251 (pre-split key, so eq.7) | eq7 0.944251 |
| `runs/leather_r50` | 1.0, pixel 0.9891957633 | 1.0, pixel 0.9891957633 |

`runs/screw_r50` predates the plain/eq7 split, so its `image_auroc` key holds
the eq.7 number; that is why it is compared against `image_auroc_eq7`.

## Licence

Every number here is derived from MVTec AD images, which are CC BY-NC-SA.
Any artifact derived from this bank inherits that licence.
