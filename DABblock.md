# DABblock.md

This file explains the DAB block components integrated into the LSNet backbone (`model/lsnet.py`), replacing the original `LSConv` (LS block) in odd-depth blocks of stages 0–2.

## Background

The original LSNet used `LSConv` (combining `LKP` large-kernel path + `SKA` Spatial Kernel Attention) as the mixer for odd-depth blocks. These have been replaced with a modified `DABModule` from [DABNet](https://arxiv.org/abs/1907.11357), adapted with larger asymmetric kernels (7×1 / 1×7) for a wider receptive field.

---

## New Classes Added

### `BNPReLU` (lines 211–220)

A small normalization + activation block: `BatchNorm2d → PReLU`.

Used internally by `Conv_DAB` and `DABModule_7` to normalize feature maps after each convolution in the DAB branch.

```
input → BN (eps=1e-3) → PReLU → output
```

---

### `Conv_DAB` (lines 223–237)

A thin wrapper around `nn.Conv2d` with an optional `BNPReLU` activation appended.

| Parameter | Meaning |
|-----------|---------|
| `nIn / nOut` | Input / output channels |
| `kSize` | Kernel size — can be `int` or `(H, W)` tuple for asymmetric kernels |
| `padding` | Must be set manually to preserve spatial size (no `padding='same'`) |
| `dilation` | `(dH, dW)` tuple — used for dilated branches |
| `groups` | Set to `nIn // 2` for depth-wise convolutions in DAB branches |
| `bn_acti` | If `True`, appends `BNPReLU` after the conv |

---

### `DABModule_7` (lines 240–273)

The core building block. It is a modified `DABModule` from DABNet where all **3×1 / 1×3** kernels are replaced with **7×1 / 1×7**, giving a larger effective receptive field without increasing the number of parameters proportionally (asymmetric depth-wise convolutions remain cheap).

#### Internal structure

```
input
  └─► BNPReLU
        └─► conv3x3  (nIn → nIn//2, 3×3, reduces channels)
              ├── Branch 1 (non-dilated asymmetric path)
              │     dconv7x1  (7×1, depth-wise, pad=(3,0))
              │     dconv1x7  (1×7, depth-wise, pad=(0,3))
              │
              └── Branch 2 (dilated asymmetric path)
                    ddconv7x1  (7×1, depth-wise, dilation=(d,1), pad=(3d,0))
                    ddconv1x7  (1×7, depth-wise, dilation=(1,d), pad=(0,3d))

              br1 + br2
                └─► BNPReLU
                      └─► conv1x1  (nIn//2 → nIn, restores channels)

output = conv1x1_result + input  ← residual connection
```

#### Padding formula

For a kernel of size `k` (along one axis) with dilation `d`, the padding to preserve spatial size is:

```
non-dilated:  pad = (k - 1) // 2       → for k=7: pad = 3
dilated:      pad = (k - 1) // 2 * d   → for k=7, d: pad = 3*d
```

#### Parameter `d` (dilation factor)

Controls the dilation rate of the second branch (`ddconv7x1` / `ddconv1x7`). Larger `d` captures longer-range context. See [Dilation Assignment per Stage](#dilation-assignment-per-stage) below.

---

## Integration into `Block` (lines 276–297)

`Block` selects its mixer based on two conditions:

| `depth % 2` | `stage` | Mixer |
|-------------|---------|-------|
| even (0, 2, 4, …) | any | `RepVGGDW` + `SqueezeExcite` |
| odd (1, 3, 5, …) | 3 | `Attention` (relative-position bias) |
| odd (1, 3, 5, …) | 0, 1, 2 | **`DABModule_7`** ← replaced here |

The `dilation` parameter is passed from `LSNet.__init__` and forwarded to `DABModule_7(ed, d=dilation)`.

Full per-block flow:

```
x → mixer (DABModule_7 / RepVGGDW / Attention)
      → SE (SqueezeExcite or Identity)
        → FFN (pointwise expand-contract with ReLU)
          → output
```

---

## Dilation Assignment per Stage (in `LSNet.__init__`)

```python
dab_block2_dilations = [4, 4, 8, 8, 16, 16]

for i, ... in enumerate(...):
    for d in range(dpth):
        if i < 2:      # stages 0 and 1 → DAB Block 1 style
            dilation = 2
        elif i == 2:   # stage 2 → DAB Block 2 style
            dilation = dab_block2_dilations[d % 6]
        else:          # stage 3 → Attention (dilation unused)
            dilation = 2
```

### Rationale

This mirrors the two-stage design of the original DABNet:

| LSNet Stage | DABNet Equivalent | Dilation strategy |
|-------------|-------------------|-------------------|
| Stage 0 | — (no LSConv blocks in lsnet_t) | 2 |
| Stage 1 | DAB Block 1 | Fixed d=2 |
| Stage 2 | DAB Block 2 | Cycling [4, 4, 8, 8, 16, 16] |
| Stage 3 | — | Attention (unchanged) |

For `lsnet_t` with `depth=[0, 2, 8, 10]`, stage 2 has 8 blocks. Only the odd-indexed blocks (1, 3, 5, 7) use `DABModule_7`. Their dilations cycle through `dab_block2_dilations`:

| Block index `d` | Dilation assigned | Used by DABModule? |
|-----------------|-------------------|--------------------|
| 0 | 4 | No (even → RepVGGDW) |
| 1 | 4 | Yes |
| 2 | 8 | No (even) |
| 3 | 8 | Yes |
| 4 | 16 | No (even) |
| 5 | 16 | Yes |
| 6 | 4 | No (even, cycles) |
| 7 | 4 | Yes |

---

## Kernel Size Comparison: Original DABModule vs DABModule_7

| Component | Original DABNet | This implementation |
|-----------|-----------------|---------------------|
| Non-dilated rows | 3×1 | **7×1** |
| Non-dilated cols | 1×3 | **1×7** |
| Dilated rows | 3×1 (D) | **7×1 (D)** |
| Dilated cols | 1×3 (D) | **1×7 (D)** |
| Entry conv | 3×3 | 3×3 (unchanged) |
| Exit conv | 1×1 | 1×1 (unchanged) |

The larger 7×1 / 1×7 kernels expand the non-dilated receptive field from 3 to 7 pixels per axis, helping the model capture wider spatial context — beneficial for segmenting large or irregularly shaped lesions.
