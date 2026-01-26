# Recod.ai Final Submission – Complete Inference Walkthrough
**YOLO + DINOv2 (ViT) + UPerNet + Robust Post-Processing**

This document explains **every single step** performed in my final submission
notebook for the Recod.ai scientific image forgery localization competition.

The explanation strictly follows the **actual execution order** of the code,
from image loading to `submission.csv`, and includes **visual Mermaid diagrams**
to make each stage intuitive.

---

## 1. Problem statement (what we are solving)

Given an input scientific image, predict:
- either `"authentic"` if no manipulation exists, or
- an **RLE-encoded pixel-level mask** of forged regions.

Challenges:
- Images may already be **cropped panels**
- Some images contain **multiple sub-panels**
- Forged regions are often **small and low-contrast**
- Naive segmentation produces many false positives

---

## 2. High-level strategy (one sentence)

> Use YOLO to decide *where to look*, DINOv2 + UPerNet to decide *what is forged*,
and disciplined post-processing to decide *whether the prediction is trustworthy*.

---

## 3. End-to-end inference pipeline

```mermaid
flowchart TD
    A[Input Image] --> B[Load RGB image<br/>Get W,H]
    B --> C[YOLO Panel Detection]
    C --> D{Boxes <= 1?}

    D -->|Yes| E[Segment Full Image]
    D -->|No| F[Crop each YOLO box<br/>with margin]

    F --> G[Segment Crop<br/>CLAHE + TTA]
    G --> H[Paste mask into global canvas<br/>Accumulate probabilities]

    E --> I[Probability map + mask]
    H --> I

    I --> J[Union masks across TTA + crops]
    J --> K[Connected Components Filter<br/>min_cc_area]
    K --> L[Compute area_metric<br/>px or ratio]
    L --> M[Compute mean_inside<br/>probability]
    M --> N{Pass thresholds?}

    N -->|No| O[Discard mask<br/>-> authentic]
    N -->|Yes| P[RLE Encode mask]

    O --> Q[submission.csv]
    P --> Q
```

## 4. Environment & setup (notebook preamble)

- Ultralytics YOLO installed offline (Kaggle-safe)
- PyTorch + HuggingFace Transformers
- OpenCV for image processing
- Matplotlib only for debug visualization

The notebook automatically switches between:

- Submission mode (test images → CSV only)
- Debug mode (train/supp images → visualize + score)

## 5. Core Configuration & Calibration

#### Global parameters

- `img_size = 532` → segmentation input resolution
- `margin = 20` → shrink YOLO boxes inward


#### Model-specific post-processing calibration

Each trained checkpoint has its own tuned parameters:

- `area_thres`
- `mean_in_thres`
- `min_cc_area`
- (optional) `use_area_ratio = True`

This avoids instability caused by different confidence distributions.


## 6. Preprocessing – CLAHE (contrast enhancement)

Before segmentation, each image (or crop) is enhanced using CLAHE:

- Convert RGB → LAB
- Apply CLAHE on L channel
- Convert back to RGB

Why

- Enhances subtle manipulation artifacts
- Improves robustness after resizing and normalization


## 7. Segmentation inference with Test-Time Augmentation (TTA)

For every image or crop, segmentation runs with three TTAs:

- no flip
- horizontal flip
- vertical flip

For each TTA:

Apply flip

1. Forward pass through model
2. Apply sigmoid(logits)
3. Unflip probability map
4. Convert to binary mask

Merged outputs:

- **Union of binary masks** → improves recall
- **Max probability map** → used for confidence gating


## 8. YOLO-guided cropping logic (critical insight)

```mermaid
flowchart LR
    A[YOLO predictions] --> B{Boxes <= 1?}
    B -- Yes --> C[Assume already a panel<br/>Segment full image]
    B -- No --> D[Crop each box<br/>Shrink by margin<br/>Segment separately]
```

Why this matters

- Many dataset images are already cropped
- Cropping again removes forged regions
- This single rule improved stability more than changing architectures


## 9. Adaptive probability → mask conversion

Instead of fixed thresholding, an adaptive, gradient-aware method is used.

```mermaid
flowchart LR
    A[Raw probability map] --> B[Sobel gradients]
    B --> C[Gradient magnitude]
    A --> D[Blend prob + gradient]
    C --> D
    D --> E[Gaussian blur]
    E --> F[Threshold = mean + 0.3*std]
    F --> G[Binary mask]
```

Intuition

- Probability → confidence
- Gradients → boundaries
- Combining both produces cleaner masks


## 10. Morphological cleanup

After thresholding:

```mermaid
flowchart LR
    A[Binary mask] --> B[Morph CLOSE<br/>5×5 kernel]
    B --> C[Morph OPEN<br/>3×3 kernel]
```

- **CLOSE** fills small holes
- **OPEN** removes tiny speckles


## 11. Crop merging & global connected components

If multiple crops exist:

- Each crop mask is pasted back to original coordinates
- Overlapping probabilities are averaged
- Binary masks are unioned

Then once globally:

```mermaid
flowchart LR
    A[Union mask] --> B[Connected Components]
    B --> C[Remove blobs < min_cc_area]
```

This avoids fragmented predictions and noise.


## 12. Final authenticity gating (most important step)

A predicted mask is accepted only if both conditions pass:

1. **Area metric**

- Pixel count OR
- Area ratio (if enabled)

2. **Mean probability inside mask**

- Average confidence of predicted region


```mermaid
flowchart LR
    A[Final mask] --> B[area_metric]
    A --> C[mean_inside]
    B --> D{>= area_thres?}
    C --> E{>= mean_in_thres?}
    D --> F{AND}
    E --> F
    F -- Yes --> G[Keep mask]
    F -- No --> H[Discard → authentic]
```

This prevents:

- Small noisy blobs
- Large but low-confidence regions


## 13. Output formatting

- Non-empty mask → Run-Length Encoding (RLE)
- Empty mask → "authentic"

Saved as:

```
case_id,annotation
```

in **submission.csv.**



## 14. Model architecture – DINOv2 + UPerNet

### Backbone: DINOv2 (Vision Transformer)

- Patch size: 14×14
- Extract intermediate features from transformer blocks:
(2, 5, 8, 11)
- CLS token removed
- Tokens reshaped into spatial feature maps

### Decoder: UPerNet

- PSP (Pyramid Scene Parsing) module for global context
- FPN-style (Feature Pyramid Networks) top-down fusion
- Multi-scale aggregation
- 1×1 classifier → segmentation logits


## 15. UPerNet architecture (visual)

## 15. UPerNet architecture (visual)

<p align="center">
  <img src="assets/recodai-yolo-dinov2-inference-seg-model.png" width="900"/>
  <br/>
  <em>DINOv2 backbone with UPerNet decode head (PSP + FPN).</em>
</p>


```mermaid
flowchart TB
    I[Input Image] --> D[DINOv2 Backbone]

    D --> C2[Feat block 2]
    D --> C3[Feat block 5]
    D --> C4[Feat block 8]
    D --> C5[Feat block 11]

    C2 --> L2[Lateral 1x1]
    C3 --> L3[Lateral 1x1]
    C4 --> L4[Lateral 1x1]
    C5 --> L5[Lateral 1x1]

    L5 --> PSP[PSP Module<br/>pools 1,2,3,6]

    PSP --> U4[Upsample + Add L4]
    U4 --> U3[Upsample + Add L3]
    U3 --> U2[Upsample + Add L2]

    U2 --> F2[FPN Conv]
    U3 --> F3[FPN Conv]
    U4 --> F4[FPN Conv]
    PSP --> F5[FPN Conv]

    F2 --> CAT[Concat all levels]
    F3 --> CAT
    F4 --> CAT
    F5 --> CAT

    CAT --> FUSE[Fuse Conv]
    FUSE --> CLS[1x1 Classifier]
    CLS --> OUT[Segmentation logits]
```


## 16. Key takeaways

- Hybrid pipelines beat monolithic segmentation
- YOLO is used for spatial reasoning, not forgery detection
- Post-processing is as important as the model
- Adaptive thresholds outperform fixed ones
- Strong gating prevents leaderboard instability
- **Public leaderboard: score: 0.312 | Rank: 665/1564**