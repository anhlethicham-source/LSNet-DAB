_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/isic_2018.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['model.lsnet', 'model.lsnet_fpn', 'mmseg_custom.model.loss_function'],
    allow_failed_imports=False
)
# model settings
model = dict(
    pretrained=None,
    type='EncoderDecoder',
    backbone=dict(
        type='lsnet_t',
        style='pytorch',
        pretrained= 'pretrain/lsnet_t.pth',
        frozen_stages=-1,
    ),
    neck=dict(
        type='LSNetFPN',
        in_channels=[64, 128, 256, 384],
        out_channels=256,
        num_outs=4
        ),
    decode_head=dict(
        out_channels=1,
        num_classes=1,
        threshold=0.5,
        loss_decode=dict(
            type='CombinedSegLoss',
            wf=0.4,           # focal — global class imbalance
            wd=0.3,           # dice  — tối ưu DSC trực tiếp
            wm=0.5,           # marginal dice — học viền để tăng IoU
            gamma=2.0,
            alpha=0.25,
            weight_in=3,
            weight_out=1,
            weight_margin=6,
            kernel_size=9,
            smooth=1.0,
            loss_name='loss_combined'
        )
    ),

    auxiliary_head=None
    )


# Use 1 for single-GPU or set to actual GPU count multiplier if scaling LR/iters
# gpu_multiples was 0 causing division by zero; set to 1 by default
gpu_multiples = 1
# optimizer — DAB layers (gn_relu, dconv, ddconv, conv3x3, conv1x1 inside mixer)
# get 5x higher LR since they are randomly initialized
optimizer = dict(
    type='AdamW',
    lr=0.0001 * gpu_multiples,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),   # pretrained backbone: 1e-5
            'backbone.blocks1.block_1.mixer': dict(lr_mult=1.0),  # DAB stage1
            'backbone.blocks2.block_1.mixer': dict(lr_mult=1.0),  # DAB stage2
            'backbone.blocks2.block_3.mixer': dict(lr_mult=1.0),
            'backbone.blocks2.block_5.mixer': dict(lr_mult=1.0),
            'backbone.blocks2.block_7.mixer': dict(lr_mult=1.0),
            'backbone.blocks3.block_1.mixer': dict(lr_mult=1.0),  # DAB stage3
            'backbone.blocks3.block_3.mixer': dict(lr_mult=1.0),
            'backbone.blocks3.block_5.mixer': dict(lr_mult=1.0),
            'backbone.blocks3.block_7.mixer': dict(lr_mult=1.0),
        }
    )
)
optimizer_config = dict()
# learning policy — warmup 2k iters để DAB layers ổn định trước khi LR cao
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.01,
    min_lr=1e-6,
    by_epoch=False
)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=4000 // gpu_multiples)
evaluation = dict(interval=4000 // gpu_multiples, metric='mIoU', save_best='Dice', rule='greater')

# Mixed precision — giảm ~40% VRAM, tăng tốc độ training trên RTX 2050 (4GB)
fp16 = dict(loss_scale='dynamic')

# Load custom dataset/model plugins from mmseg_custom
plugin = True
plugin_dir = 'mmseg_custom'
