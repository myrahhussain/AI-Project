# Interim demo (Tue Nov 24) – what to show

1. Problem + plan (README, 1 slide): stain shift across hospitals; one ResNet-18; baseline vs HED vs Vahadane-style vs both.
2. Implementation progress: repo structure, 43 passing tests, leakage-safe splits (official id_val shares all 30 slides with train, so we hold out whole slides ourselves).
3. Figures: `results/aug_grid*.png` (HED strengths), normalisation before/after on real patches.
4. Initial results: table from `evaluate.summary_table` (mean ± std over seeds), ID vs unseen hospital 1. Test hospital 2 is NOT used yet.
5. Honest limits: small subset, few seeds, results preliminary; CAM failure analysis and equivariant model are for the final.

Only claim what the printed table shows.
