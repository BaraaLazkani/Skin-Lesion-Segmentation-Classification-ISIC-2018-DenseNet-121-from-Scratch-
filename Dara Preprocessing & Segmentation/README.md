# Phase 1 — Preprocessing & Segmentation

> **Attribution:** this phase was provided as part of the project framework and is **not** original work of the authors. It is documented here for completeness and to explain the data format consumed by [Phase 2 (Training and Validation)](../Training%20and%20Validation/README.md).

## What this directory does

Raw ISIC 2018 dermoscopy JPEGs are noisy: uneven illumination, hair occlusion, and irrelevant background skin all interfere with lesion classification. This phase turns each raw image into a clean, segmented lesion crop before any learning happens.

```
Dara Preprocessing & Segmentation/
├── src/
│   └── 01_preprocessing_and_segmentation.ipynb   # the full pipeline, notebook form
└── results/
    ├── ISIC2018_Task3_Training_GroundTruth.csv   # one-hot labels, training split
    ├── ISIC2018_Task3_Test_GroundTruth.csv       # one-hot labels, test split
    └── images/
        ├── training/   # 8,901 segmented training JPEGs
        └── testing/    # 1,512 segmented test JPEGs
```

## Pipeline

Each raw image passes through the following stages, in order:

1. **Homomorphic filtering** — normalises non-uniform illumination by suppressing the low-frequency illumination component in the log-frequency domain while preserving high-frequency reflectance detail.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — local histogram equalisation with a clip limit, improving visibility of fine lesion structure without amplifying noise.
3. **Grayscale conversion** — feeds the hair-detection branch.
4. **Black Hat morphological filtering** — detects dark, elongated structures (hair fibres) against brighter skin by subtracting a morphologically closed image from the original.
5. **Gaussian filtering** — smooths the hair-response map to suppress spurious detections.
6. **Adaptive thresholding** — binarises the smoothed response into a hair mask, threshold adapting to local neighbourhood statistics.
7. **Inpainting (hair removal)** — the (slightly dilated) hair mask is used to fill hair pixels from surrounding context, producing a clean skin surface.
8. **Resize to 256×256** — bicubic interpolation, chosen over bilinear for its better edge preservation at this scale.
9. **Fuzzy C-Means (FCM) segmentation** — pixels are grouped into `k=2` fuzzy clusters (lesion ROI vs. background skin). Fuzziness exponent `m = 0.001`, tolerance `ε = 1e-5`, max 1,000 iterations. The ROI cluster becomes the binary segmentation mask.
10. **Post-processing** — morphological closing fills small holes in the mask and smooths its boundary.

## Output format

| Split | Filename pattern | Label source |
|---|---|---|
| Training | `ISIC_XXXXXXX_segmented.jpg` | `ISIC2018_Task3_Training_GroundTruth.csv` |
| Testing | `ISIC_XXXXXXX_Test_Segmented.jpg` | `ISIC2018_Task3_Test_GroundTruth.csv` |

Each ground-truth CSV is one-hot encoded across the seven ISIC 2018 Task 3 diagnostic classes:

| Code | Full name | Train count | Description |
|---|---|---:|---|
| MEL | Melanoma | 1,113 | Malignant melanocytic tumour |
| NV | Melanocytic Nevi | 6,705 | Common benign mole |
| BCC | Basal Cell Carcinoma | 514 | Most common skin cancer |
| AKIEC | Actinic Keratosis / Bowen's Disease | 327 | Pre-malignant sun-damage lesion |
| BKL | Benign Keratosis-like Lesion | 1,099 | Seborrheic keratosis |
| DF | Dermatofibroma | 115 | Benign fibrous skin nodule |
| VASC | Vascular Lesions | 142 | Angioma, angiokeratoma, etc. |

The dataset is heavily imbalanced — NV alone is ≈68% of the training split, while DF and VASC together are under 3%. This imbalance is the central design constraint driving the sampling and loss choices in Phase 2.

`Training and Validation/src/dataset.py` reads both the images and these CSVs directly (see `load_and_clean_csv`), so the filename suffixes and column names above are a hard contract between the two phases.
