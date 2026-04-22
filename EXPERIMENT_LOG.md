# Experiment Log — LSNet ISIC 2018

Ghi lại tất cả các phiên bản config đã thử nghiệm để có thể reset về bất kỳ phiên bản nào.

---

## V1 — Baseline gốc ✅ DSC = 89.4% (BEST)

**Git commit:** `133c6b8`  
**Branch:** main

### Kết quả
| DSC | IoU | Precision | Recall | F1 |
|-----|-----|-----------|--------|----|
| 89.4 | 81.13 | 90.12 | 88.76 | 89.4 |

### Config chính (`fpn_lsnet_isic_2018.py`)
```python
custom_imports = dict(imports=['mmseg_custom.fix_isic'], allow_failed_imports=False)

decode_head=dict(num_classes=2)   # <-- CrossEntropyLoss mặc định của MMSeg FPNHead

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
```

### Model (`model/lsnet.py`)
- Backbone: LSConv blocks (không có DAB)
- BN eval mode: **BẬT** (tất cả BatchNorm giữ eval trong training)
- Block cấu trúc odd-depth stages 0-2: `LSConv`

### Cách reset về V1
```bash
git checkout 133c6b8 -- configs/sem_fpn/fpn_lsnet_isic_2018.py
git checkout 133c6b8 -- model/lsnet.py
```

---

## V2 — FocalDiceLoss + LSConv

**Git commit:** `69904e4`  
**Branch:** func_loss_DAB

### Config thay đổi so với V1
```python
decode_head=dict(
    out_channels=1, num_classes=1,
    loss_decode=dict(
        type='FocalDiceLoss',
        gamma=2.0, alpha=0.75, wf=1.0, wd=0.5, smooth=1.0
    )
)
runner = dict(max_iters=20000)
```

### Ghi chú
- Chưa có kết quả rõ ràng (20k iter ngắn)

---

## V3 — DAB thay LSConv + FocalDiceLoss ❌ DSC = 84.29%

**Git commit:** `3cbfac0`  
**Branch:** func_loss_DAB

### Kết quả
| DSC | IoU | Precision | Recall | F1 |
|-----|-----|-----------|--------|----|
| 84.29 | 72.84 | 75.04 | 96.13 | 84.29 |

### Vấn đề chính
1. **BN eval mode bị XÓA** → BN train mode với batch=2 cực kỳ nhiễu
2. **DAB layers random init** → không có pretrained weights
3. **FocalDiceLoss wf=1, wd=1.0** — wd quá cao

### Threshold experiments trên V3
| Threshold | DSC | IoU | Precision | Recall |
|-----------|-----|-----|-----------|--------|
| 0.5 | 84.23 | 72.75 | 77.71 | 91.93 |
| 0.7 | 84.63 | 73.36 | 78.97 | 91.18 |
| 0.8 | 84.76 | 73.54 | 79.38 | 90.91 |

---

## V4 — MarginalDiceLoss + DAB ❌ Peak 44k rồi giảm

**Git commit:** `62fc390`  
**Branch:** func_loss_DAB

### Config thay đổi so với V3
```python
loss_decode=dict(
    type='MarginalDiceLoss',
    weight_in=3, weight_out=1, weight_margin=6,
    kernel_size=9, smooth=1e-3
)
runner = dict(max_iters=80000)
```

### Vấn đề
- Peak tại ~44k iterations rồi giảm liên tục → **overfitting**
- BN eval mode vẫn bị xóa

---

## V5 — Local Dice Loss + DAB ❌ Peak 4000 rồi giảm

**Branch:** func_loss_DAB (experiment không commit)

### Vấn đề
- Diverge rất sớm (4000 iter) → loss không ổn định với DAB random init

---

## V7 — DAB + GNPReLU + CombinedSegLoss + fp16 🔄 Đang thử

**Branch:** func_loss_DAB

### CombinedSegLoss = Focal (0.4) + Dice (0.3) + MarginalDice (0.5)
- MarginalDice: `weight_in=3, weight_out=1, weight_margin=6, kernel_size=9`
- fp16 mixed precision: tiết kiệm ~40% VRAM, phù hợp RTX 2050 4GB

### Mục tiêu: DSC ≥ 90%, IoU ≥ 84%

---

## V6 — DAB + GNPReLU + FocalDiceLoss + Warmup

**Branch:** func_loss_DAB (commit hiện tại)

### Thay đổi so với V4

**`model/lsnet.py`:**
- `BNPReLU` → `GNPReLU` (GroupNorm, stable với batch=2)
- Khôi phục BN eval mode trong `train()`

**`configs/sem_fpn/fpn_lsnet_isic_2018.py`:**
```python
decode_head=dict(
    out_channels=1, num_classes=1,
    threshold=0.5,
    loss_decode=dict(
        type='FocalDiceLoss',
        gamma=2.0, alpha=0.25, wf=1.0, wd=0.5, smooth=1.0
    )
)

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),              # pretrained: LR=1e-5
        'backbone.blocks2.block_1.mixer': dict(lr_mult=1.0),  # DAB stage1: LR=1e-4
        'backbone.blocks3.block_1.mixer': dict(lr_mult=1.0),  # DAB stage2
        'backbone.blocks3.block_3.mixer': dict(lr_mult=1.0),
        'backbone.blocks3.block_5.mixer': dict(lr_mult=1.0),
        'backbone.blocks3.block_7.mixer': dict(lr_mult=1.0),
    })
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear', warmup_iters=2000, warmup_ratio=0.01,
    min_lr=1e-6, by_epoch=False
)
runner = dict(max_iters=80000)
evaluation = dict(interval=4000, metric='mIoU', save_best='Dice')
```

### Mục tiêu
- DSC ≥ 90%, IoU ≥ 85%

---

## Cách reset nhanh về V1 (89.4 DSC)

```bash
# Reset config về V1
git checkout 133c6b8 -- configs/sem_fpn/fpn_lsnet_isic_2018.py

# Reset lsnet.py về V1 (LSConv, BN eval mode)
git checkout 133c6b8 -- model/lsnet.py

# Kiểm tra
python tools/train.py configs/sem_fpn/fpn_lsnet_isic_2018.py --work-dir ./work_dirs/v1_baseline
```

**Lưu ý V1:** `custom_imports` dùng `mmseg_custom.fix_isic`, không phải `mmseg_custom.model.loss_function`.
