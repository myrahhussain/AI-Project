# Results (Widad's final runs, 28 Nov 2026)

Produced with `notebooks/final_experiments_solo.ipynb` on Colab (T4). Model checkpoints
(2.4 GB) are not in the repo; they stay on Google Drive.

| Folder | What | Made by |
|---|---|---|
| `runs/` | one JSON per training run: `<tag>_<method>_seed<n>.json` with config, threshold, id_val and hospital-1 (8,000-patch) metrics | `scripts/run_experiments.py` |
| `summary/` | the runs above as one table (mean ± std over seeds) | `scripts/summarize.py` |
| `final/` | every patch of hospital 1 (34,904) and the sealed hospital 2 (85,054) scored by every saved model; `final_h*_summary.csv` = mean ± std per tag and method; `probs_h*/` = per-patch tumour probabilities (float16), `index.npy` = patch indices | `scripts/predict_hospital.py` |
| `size/` | metastasis size of all 50 slides measured from the pathologists' tumour patches, vs CAMELYON17's labels | `scripts/metastasis_size.py` |
| `size_models/` | the same measured from each model's tumour calls on hospitals 1 and 2 | `scripts/metastasis_size.py --final` |

Tags: `main` (ImageNet ResNet-18, 5 seeds), `bb_r18histo` (Ciga et al. 2022), `bb_r50histo`
(Kang et al. 2023). Still to come: `abl_sigma*`, `abl_ref_*`, `loho_h*`.
