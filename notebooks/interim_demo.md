# Interim demo (Tue Nov 24): what to show

Aim: show a working pipeline, real numbers for all four methods and an honest reading of them.
The test hospital (2) has NOT been used. Say so.

1. **Problem and plan (1 slide).** Stain differs across hospitals; one ResNet-18; baseline vs HED
   augmentation vs Vahadane-style normalization vs both. README has the table of methods.
2. **Data and splits.** Train: hospitals 0, 3, 4 (24 slides, 40k patches). id_val: 6 held-out slides of the
   same hospitals. ood_val: hospital 1 (10 slides). Test: hospital 2, untouched. The official id_val shares
   all slides with train, so we hold out whole slides ourselves (leakage check in `tests/test_splits.py`).
3. **Figures** (from `figures_v3/`):
   - `hed_augmentation.png`: what the augmentation does at each strength.
   - `normalization_train.png`, `normalization_ood_val.png`: before/after on real patches.
   - `results_unseen_hospital.png`: accuracy and AUC on hospital 1 (y-axis starts at 0).
4. **Results** (hospital 1, mean ± std over 5 seeds):

   | Method | Accuracy | Sens. | Spec. | AUC |
   |---|---|---|---|---|
   | baseline | 0.852 ± 0.010 | 0.959 | 0.746 | 0.959 |
   | hed | 0.864 ± 0.005 | 0.935 | 0.794 | 0.956 |
   | vahadane | 0.884 ± 0.004 | 0.839 | 0.930 | 0.954 |
   | both | 0.872 ± 0.012 | 0.786 | 0.958 | 0.946 |

5. **What the numbers say.**
   - Normalization helps accuracy most (about +3 points); HED a little; both together is not better than
     normalization alone.
   - AUC does not improve. The gains come from the decision threshold (chosen on same-hospital validation)
     fitting the new hospital better: the baseline over-calls tumour there, normalization under-calls it.
   - Failure analysis: errors cluster in a few slides (worst slide about 41% error for the baseline) and in
     dense lymphocyte-like tissue, edges and blank patches. CAMs show where the model reacted and are a
     way to investigate mistakes, not proof of how it reasons.
6. **A bug we found and fixed (good to mention).** Normalization first washed out slides because a few
   percent of near-black pixels set the stain-strength estimate. We now ignore saturated pixels
   (regression test in `tests/test_normalize.py`) and reran everything.
7. **Honest limits.** Subset of the data, 5 seeds, one validation hospital, threshold chosen on one
   source. Not done yet: final test-hospital run (once, at the end), comparison with an open-source
   normalizer, the equivariant model (stretch), report and slides.

Only claim what the printed table shows.

## Questions to be ready for
- *Why did accuracy change but AUC not?* The ranking ability is the same; the threshold moves.
- *Why not tune on the test hospital?* It is the one clean check, so it is run once at the end.
- *Why hold out slides yourselves?* The official split leaks slides between train and id_val.
- *Do CAMs prove the model uses tumour features?* No; they only show where it reacted.
