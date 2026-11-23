# Robust Detection of Breast Cancer Metastases in Histopathology Images

Course project (AI). A controlled comparison of stain normalization, stain
augmentation and their combination on one baseline architecture (an
ImageNet-pretrained ResNet-18), evaluated on Camelyon17-WILDS with a hospital
held out at test time.

**Group members:** Widad Fatima Khan, Myrah Hussain, Romaisa Shazad

## Question
Hospitals stain their slides differently. A model trained on some hospitals can
get worse on a hospital it has never seen. Which cheap fix helps, and why?

Methods compared (same network, same data subset, same training budget):

| Method | What it does |
|---|---|
| `baseline` | flips and 90-degree rotations only |
| `hed` | random per-channel scaling in HED stain space during training |
| `vahadane` | Vahadane-style stain normalization to one reference (per-slide stain estimate) |
| `both` | normalization, then HED augmentation |

## Data and splits
Camelyon17-WILDS, 96x96 patches, a class-balanced subset sampled by slide.

| Split | Hospitals | Slides | Patches | Use |
|---|---|---|---|---|
| train | 0, 3, 4 | 24 | 40,000 | fit the model |
| id_val | 0, 3, 4 | 6 (held-out slides) | 8,000 | choose the decision threshold |
| ood_val | 1 | 10 | 8,000 | unseen-hospital validation |
| test | 2 | - | - | **not used yet**; one run at the very end |

The official `id_val` split shares every slide with `train`, so we hold out whole
slides ourselves (checked by `tests/test_splits.py`).

## Results so far (unseen hospital 1, mean ± std over 5 seeds)
| Method | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| baseline | 0.852 ± 0.010 | 0.959 ± 0.013 | 0.746 ± 0.031 | 0.959 ± 0.003 |
| hed | 0.864 ± 0.005 | 0.935 ± 0.015 | 0.794 ± 0.019 | 0.956 ± 0.006 |
| vahadane | 0.884 ± 0.004 | 0.839 ± 0.007 | 0.930 ± 0.007 | 0.954 ± 0.005 |
| both | 0.872 ± 0.012 | 0.786 ± 0.025 | 0.958 ± 0.007 | 0.946 ± 0.007 |

Reading these numbers (preliminary, one validation hospital, subset of the data):
- Normalization gives the clearest accuracy gain (about +3 points). HED gives a
  smaller one. Combining the two did not beat normalization alone.
- AUC does not improve for any method. The accuracy gains mostly come from the
  decision threshold (picked on same-hospital validation) landing better on the new
  hospital: the baseline over-calls tumour there, normalization under-calls it.
- Failure analysis (CAM heatmaps, used to investigate errors and not as proof of
  what the model reasons): remaining errors concentrate in a few slides and in dense
  lymphocyte-like tissue, tissue edges and blank patches; baseline and HED make
  mostly the same confident mistakes.

## Run it
Colab (GPU), see `notebooks/run_experiments.ipynb`:

```
python scripts/run_experiments.py --root /content/data --cache <cache> --out <results> \
    --n_train 40000 --epochs 5 --seeds 0 1 2 3 4 --methods baseline hed vahadane both \
    --tag v3 --ckpt_dir <ckpt>
python scripts/make_figures.py --cache <cache> --results <results> --out <figures> --tag v3
python scripts/failure_analysis.py --root /content/data --cache <cache> --ckpt_dir <ckpt> \
    --results <results> --out <out> --tag v3 --methods baseline hed
```

Tests (CPU, synthetic data): `python -m pytest tests -q`

## Status
- Done and run on real data: splits, HED augmentation, stain normalization, training
  and evaluation for all four methods (5 seeds), figures, CAM failure analysis.
- Not done yet: final test-hospital evaluation (kept untouched on purpose),
  comparison of our normalizer against an open-source implementation, the
  equivariant model (stretch goal), report and final slides.
