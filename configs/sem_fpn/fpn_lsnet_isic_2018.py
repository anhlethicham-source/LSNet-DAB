_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/isic_2018.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(imports=['mmseg_custom.fix_isic'], allow_failed_imports=False)
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
        num_classes= 1,
        loss_decode=dict(
            type='FocalDiceLoss',
            gamma=2.0,
            alpha=0.75,
            wf=1,
            wd=0.5,
            smooth=1.0,
            loss_name='loss_focal_dice'
        )
    ),
            

    auxiliary_head=None
    )


# Use 1 for single-GPU or set to actual GPU count multiplier if scaling LR/iters
# gpu_multiples was 0 causing division by zero; set to 1 by default
gpu_multiples = 1
# optimizer
optimizer = dict(type='AdamW', lr=0.0001 * gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=4000 // gpu_multiples)
evaluation = dict(interval=4000 // gpu_multiples, metric='mIoU')

# Load custom dataset/model plugins from mmseg_custom
plugin = True
plugin_dir = 'mmseg_custom'
