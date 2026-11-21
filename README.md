# Robust Detection of Breast Cancer Metastases in Histopathology Images

Course project (AI). A controlled comparison of stain normalization, stain
augmentation and their combination on one baseline architecture (ResNet-18),
evaluated on Camelyon17-WILDS with leave-one-hospital-out testing.

**Group members:** Widad Fatima Khan, Myrah Hussain, Romaisa Shazad

## Status
Project setup. No experiments have been run yet.

## Plan
1. Load Camelyon17-WILDS and build slide-separated splits
2. Baseline ResNet-18 with flips and 90-degree rotations only
3. HED stain augmentation
4. Vahadane-style stain normalization (validated against an open-source implementation)
5. Combined method, per-hospital sensitivity/specificity and gap reporting
6. Class activation maps for failure analysis
