norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='lsnet_t',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        pretrained='pretrain/lsnet_t.pth',
        frozen_stages=-1),
    neck=dict(
        type='LSNetFPN',
        in_channels=[64, 128, 256, 384],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='FocalDiceLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            gamma=2.0,
            alpha=0.25,
            wf=0.5,
            wd=0.5,
            smooth=1.0,
            loss_name='loss_focal_dice'),
        out_channels=1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    auxiliary_head=None)
dataset_type = 'ISIC2018Dataset'
data_root = '/mnt/d/imagenet/data/'
data_mask_root = '/mnt/d/imagenet_processed/masks'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=0),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='ISIC2018Dataset',
            data_root='/mnt/d/imagenet/data/',
            img_dir='images/train',
            ann_dir='/mnt/d/imagenet_processed/masks/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(256, 256),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.5, direction='vertical'),
                dict(
                    type='RandomRotate',
                    prob=0.5,
                    degree=180,
                    pad_val=0,
                    seg_pad_val=0),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=0),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='ISIC2018Dataset',
        data_root='/mnt/d/imagenet/data/',
        img_dir='images/val',
        ann_dir='/mnt/d/imagenet_processed/masks/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ISIC2018Dataset',
        data_root='/mnt/d/imagenet/data/',
        img_dir='images/test',
        ann_dir='/mnt/d/imagenet_processed/masks/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
custom_imports = dict(
    imports=[
        'model.lsnet', 'model.lsnet_fpn', 'mmseg_custom.model.loss_function'
    ],
    allow_failed_imports=False)
gpu_multiples = 1
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-06, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
plugin = True
plugin_dir = 'mmseg_custom'
