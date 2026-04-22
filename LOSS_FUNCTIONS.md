# Loss Functions — Detailed Reference

All loss classes live in `mmseg_custom/model/loss_function/dice_function.py` and are registered into MMSegmentation's `LOSSES` registry.

---

## Active Loss: `CombinedSegLoss`

**Config:** `configs/sem_fpn/fpn_lsnet_isic_2018.py` → `loss_decode`

```python
dict(
    type='CombinedSegLoss',
    wf=0.4, wd=0.3, wm=0.5,
    gamma=2.0, alpha=0.25,
    weight_in=3, weight_out=1, weight_margin=6,
    kernel_size=9, smooth=1.0
)
```

### Motivation

Binary medical segmentation (skin lesion vs. background) has two core difficulties:

1. **Class imbalance** — lesion pixels are typically a small fraction of the image.
2. **Boundary precision** — clinical metrics (Dice, IoU) are highly sensitive to boundary quality; a few mis-classified boundary pixels disproportionately hurt the score.

`CombinedSegLoss` attacks both problems simultaneously with three complementary terms.

---

### Total Loss Formula

```
L = wf * L_focal  +  wd * L_dice  +  wm * L_marginal
  = 0.4 * L_focal  +  0.3 * L_dice  +  0.5 * L_marginal
```

---

### Component 1 — Focal Loss (`L_focal`, weight = 0.4)

**Formula:**

```
p      = sigmoid(logit)
p_t    = p  · y  +  (1 - p) · (1 - y)          # confidence for the correct class
L_focal = mean[ α · (1 - p_t)^γ · BCE(logit, y) ]
```

- `α = 0.25` — down-weights the majority class (background).
- `γ = 2.0` — modulating factor; when `p_t → 1` (easy example) the loss → 0, so training focuses on hard pixels.

**Why it works:**  
Standard BCE treats every pixel equally. In a typical skin lesion image, ~80-90% of pixels are background — a network that always predicts background would have low BCE but be useless. Focal loss suppresses the overwhelming easy-background gradient and forces the network to learn the lesion region.

**Effectiveness:**  
- Prevents the model from collapsing to the "all-background" trivial solution.
- Improves recall on small or low-contrast lesions.
- Well-established in object detection (RetinaNet) and adapted here for pixel-wise segmentation.

---

### Component 2 — Standard Dice Loss (`L_dice`, weight = 0.3)

**Formula:**

```
P_flat = sigmoid(logit).flatten(spatial)    # shape (B, H*W)
T_flat = target.flatten(spatial)

L_dice = 1 - mean_batch[ (2 · Σ(P·T) + smooth) / (Σ P + Σ T + smooth) ]
```

- `smooth = 1.0` prevents division by zero and stabilises gradients for nearly-empty masks.

**Why it works:**  
Dice loss directly optimises the Dice Similarity Coefficient (DSC), which is the primary evaluation metric for medical segmentation. Cross-entropy optimises per-pixel log-likelihood; it does not directly maximise DSC. Jointly using both aligns the training objective with the evaluation metric.

**Effectiveness:**  
- Gives the model a direct gradient signal toward maximising overlap.
- Particularly helpful when lesions are small, because each true-positive pixel contributes more to the Dice numerator than to BCE.
- Complementary to Focal: Focal fixes the imbalance problem; Dice fixes the metric-alignment problem.

---

### Component 3 — Marginal Dice Loss (`L_marginal`, weight = 0.5)

Inspired by ResMamba-ULite. This is the highest-weighted component, reflecting its importance for boundary quality.

#### Step 1 — Build a Spatial Weight Map (no gradient, from GT only)

```
dilated(x) = clamp( conv2d(T, ones_9x9),  0, 1 )    # expand mask outward
eroded(x)  = floor( conv2d(T, ones_9x9 / 81) + 1e-2 ) # shrink mask inward

W(x) = (dilated - eroded) * weight_margin    # boundary zone  → weight = 6
      + eroded             * weight_in        # interior zone  → weight = 3
      + (1 - dilated)      * weight_out       # background zone → weight = 1
```

| Zone | Pixels | Weight |
|------|--------|--------|
| Interior (eroded region) | Deep inside lesion | 3 |
| **Boundary** (dilated − eroded) | Ring around lesion edge | **6** |
| Background (outside dilated) | Far from lesion | 1 |

The boundary zone receives 6× the weight of the background — errors there are penalised the most.

#### Step 2 — Weighted Dice

```
L_marginal = mean_batch[
    1 - (2 · Σ(W·P·T) + 1e-3) / (Σ(W·P) + Σ(W·T) + 1e-3)
]
```

**Why it works:**  
Standard Dice (and BCE) weight all pixels equally. Mis-segmenting a boundary pixel hurts IoU just as much as mis-segmenting an interior pixel, yet the network sees no extra reason to be careful at boundaries. By multiplying the Dice numerator and denominator by the weight map, boundary pixels contribute disproportionately to the loss — the gradient pushes the network to get the boundary right.

**Effectiveness:**  
- Directly improves IoU and Hausdorff distance metrics.
- Captures fine-grained lesion contours that matter in clinical assessment.
- Weight map is computed without gradient from GT, so no additional learnable parameters.

---

### Interaction Between the Three Components

```
Focal      → "learn to see the lesion at all" (global class balance)
Dice       → "maximise overlap with GT" (metric alignment)
MarginalDice → "get the boundary right" (spatial precision)
```

The weights `wf=0.4, wd=0.3, wm=0.5` reflect empirical priority:
- MarginalDice gets the highest weight because boundary quality is the hardest and most impactful.
- Focal gets second priority to maintain class-imbalance correction.
- Dice acts as a regulariser that keeps the overall overlap from drifting.

---

## Other Available Losses

### `FocalDiceLoss`

```
L = wf * L_focal  +  wd * L_dice     (default: 0.5 / 0.5)
```

A lighter predecessor to `CombinedSegLoss`. Handles class imbalance and metric alignment but has no boundary awareness. Suitable when the dataset has clear, well-defined boundaries and the main challenge is class imbalance.

**Limitations vs. CombinedSegLoss:**  
Treats boundary and interior pixels identically → weaker IoU on datasets with irregular or fuzzy lesion edges.

---

### `MyBCEDiceLoss`

```
L = wb * L_bce  +  wd * L_dice     (default: 0.5 / 0.5)
```

Replaces Focal with plain BCE. Supports an optional `pos_weight` scalar to handle imbalance in BCE. Simpler and more stable to tune. Good baseline for moderately imbalanced datasets. Lacks the hard-example mining of Focal and any boundary awareness.

---

### `MarginalDiceLoss`

Standalone boundary-aware Dice (the marginal component extracted for independent use). Useful when combined with an external BCE/Focal term, or for ablation studies isolating the boundary contribution.

---

## Summary Comparison

| Loss | Handles Imbalance | Metric-Aligned | Boundary-Aware | Complexity | Status |
|---|---|---|---|---|---|
| `CombinedSegLoss` | Focal | Dice + MarginalDice | Yes (6× boundary) | High | **Active** |
| `FocalDiceLoss` | Focal | Dice | No | Medium | Available |
| `MyBCEDiceLoss` | BCE pos_weight | Dice | No | Low | Available |
| `MarginalDiceLoss` | No | MarginalDice | Yes | Low | Available |

---

## Hyperparameter Sensitivity Notes

| Parameter | Effect | Recommended range |
|---|---|---|
| `gamma` (Focal) | Higher → harder focus; too high causes instability | 1.5–3.0 |
| `alpha` (Focal) | Lower → more weight on foreground; tune with class ratio | 0.25–0.75 |
| `weight_margin` | Higher → stronger boundary signal; too high ignores interior | 4–10 |
| `kernel_size` (morph) | Larger → wider boundary zone | 7–13 (odd) |
| `smooth` (Dice) | Higher → more stable on nearly-empty masks | 0.1–10.0 |
| `wf / wd / wm` | Relative emphasis; wm > wd suggested for IoU-heavy evals | sum ≈ 1.0–1.5 |
