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
| test | 2 | 10 | 85,054 (all) | sealed; scored once on 28 Nov 2026 with every setting frozen |

The official `id_val` split shares every slide with `train`, so we hold out whole
slides ourselves (checked by `tests/test_splits.py`).

## Final results (main ResNet-18, all patches, mean ± std over 5 seeds)
| Method | Hosp. 1 accuracy | Hosp. 1 AUC | Hosp. 2 accuracy | Hosp. 2 sensitivity | Hosp. 2 AUC | Hosp. 2 drop won back |
|---|---|---|---|---|---|---|
| baseline | 0.868 ± 0.014 | 0.977 | 0.735 ± 0.032 | 0.471 | 0.936 | – |
| hed | 0.887 ± 0.007 | 0.976 | 0.783 ± 0.019 | 0.568 | 0.958 | 22% |
| vahadane | 0.914 ± 0.004 | 0.974 | 0.939 ± 0.007 | 0.890 | 0.983 | 91% |
| both | 0.906 ± 0.010 | 0.971 | 0.953 ± 0.005 | 0.919 | 0.990 | 96% |

- On the sealed hospital 2 the baseline misses about half the tumours; normalisation wins back
  91–96% of the accuracy drop and raises AUC, so the hypothesis (a fix closes at least half the
  gap) holds there. On hospital 1 the gain comes mostly from the threshold (AUC flat).
- Pathology-pretrained backbones (`resnet18_histo`, `resnet50_histo`) did not generalise better
  than the ImageNet ResNet-18; with normalisation all three reach 0.939–0.965 on hospital 2.
- Metastasis size (ITC / micro / macro) measured from patches matches CAMELYON17's slide labels
  on 44 of 50 slides. On hospital 2 the baseline would under-stage 5 of 10 slides; normalisation
  about 1.
- Pending: σ and reference ablations, leave-one-hospital-out.

Every result file is in `results/` (see `results/README.md`).

## Interim results (unseen hospital 1, 8,000-patch sample, mean ± std over 5 seeds)
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

## Final experiments (for the final report)
Run `notebooks/final_experiments.ipynb` top to bottom (Colab, T4). Every run is resumable.
Settings are frozen (σ = 0.2, pooled reference, 5 epochs, 40,000 patches, threshold from id_val).

| Part | What | Script |
|---|---|---|
| Ablations | HED σ = 0.05 / 0.1 / 0.4; normalisation reference from one lab (h0, h3, h4) | `run_experiments.py --sigma`, `--ref` |
| Backbones | `resnet18_histo` (Ciga et al. 2022), `resnet50_histo` (Kang et al. 2023), optional `convnext_tiny` | `run_experiments.py --backbone --weights_path` |
| Final test | every patch of hospital 1, then the one-time test on hospital 2 | `predict_hospital.py` |
| Leave-one-hospital-out | train on four hospitals, test on the fifth, for all five | `run_experiments.py --split loho --test_hospital H --unseal_test` |
| Metastasis size | ITC / micro / macro per slide from tumour calls, vs CAMELYON17 slide labels | `metastasis_size.py` |
| Tables | everything above as small CSV / markdown tables | `summarize.py` |

Hospital 2 can only be loaded with `--unseal_test`. Histology weights are downloaded by the
notebook (URLs in `src/models.py`) and read without executing anything stored in the file.

## Status (28 Nov 2026)
- Done on real data: main study rerun from scratch (matches the interim exactly), histology
  backbones, scoring of every hospital-1 patch, the one-time hospital-2 test, metastasis size.
- Running: σ and reference ablations, leave-one-hospital-out.
- Dropped: the compact CNN trained from scratch (replaced by the two published backbones) and
  the equivariant model (stretch goal), for time.
