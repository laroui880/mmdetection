_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/fire_smoke_detection.py', './rtmdet_tta.py'
]
# ==============Custom Variables==============
# -----runtime related-----

checkpoint = '/hotdata/userdata/sarah.laroui/workspace/mmdetection/workdir/finetune_azuria-smoke-v2/ssl_patternnet_smk_frozen1_rtmdet_tiny_syncbn_fast_4xb4-3000e_smoke_detection/best_coco/bbox_mAP_epoch_620.pth'

env_cfg = dict(cudnn_benchmark=True)
workflow = [('train', 1), ('val', 1)]
# -----data related-----
img_scale = _base_.img_scale
num_classes = _base_.num_classes
# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mixup
mixup_max_cached_images = 10
# Batch size
batch_size = _base_.batch_size
# Number of workers
num_workers = _base_.num_workers

max_epochs = 100
stage2_num_epochs = 20
base_lr = 0.004
interval = 10
# -----train val related-----
lr_start_factor = 1.0e-5
weight_decay = 0.05 #TODO: to understand
max_keep_ckpts=3 # only keep latest 3 checkpoints
val_interval=2 # number of epochs interval to do validation during training

# -----model related-----
deepen_factor = 0.167
widen_factor= 0.375
norm_cfg = dict(type='BN')  # Normalization config
strides = [8, 16, 32]
mean = _base_.mean
std = _base_.std
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss
nms_iou = 0.65
# -----save train data-----
#work_dir = f"/trainings/rtmdet_tiny_syncbn_fast_{num_workers}xb{batch_size}-{max_epochs}e_smoke-v2"
work_dir = f"/hotdata/userdata/sarah.laroui/workspace/mmdetection/workdir/test_fire_smoke/rtmdet_tiny_syncbn_fast_{num_workers}xb{batch_size}-{max_epochs}e_fallen_person_azuria"


#=============================================
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu'
        ),
        frozen_stages=1),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=num_classes,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight),
        loss_bbox=dict(type='GIoULoss', loss_weight=loss_bbox_weight),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=nms_iou),
        max_per_img=300),
)

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='RandomResize',
    #     scale=(img_scale[0] * 2, img_scale[1] * 2),
    #     ratio_range=random_resize_ratio_range,
    #     keep_ratio=True),

    dict(
        type='Resize',
        scale_factor=1.0,
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, crop_type='absolute'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    # dict(
    #     type='CachedMixUp',
    #     img_scale=img_scale,
    #     ratio_range=(1.0, 1.0), # TODO: search the value for this ratio range (not provided in mmyolo config)
    #     max_cached_images=mixup_max_cached_images,
    #     pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='RandomResize',
    #     scale=img_scale,
    #     ratio_range=random_resize_ratio_range,
    #     keep_ratio=True),
    dict(
        type='Resize',
        scale_factor=1.0,
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, crop_type='absolute'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args={{_base_.file_client_args}}),
    
    # dict(
    #     type='Resize',
    #     scale_factor=1.0,
    #     keep_ratio=True),

    # dict(type='RandomCrop', crop_size=img_scale, crop_type='absolute'),

    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    

    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataloaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# Evaluators are used to compute the metrics of the trained model on the validation and testing datasets. 
# The config of evaluators consists of one or a list of metric configs:
val_evaluator = dict(proposal_nums=(100, 1, 10)) # https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.evaluation.metrics.CocoMetric 
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# param_scheduler is a field that configures methods of adjusting optimization hyperparameters such as learning rate and momentum. 
# Users can combine multiple schedulers to create a desired parameter adjustment strategy.
# Find more in parameter scheduler tutorial and parameter scheduler API documents
param_scheduler = [
    dict(
        type='LinearLR', #TODO: search how it works
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=max_epochs),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'  
    ),
    visualization=dict(draw=True, interval=max_epochs-1), # https://mmdetection.readthedocs.io/en/3.x/api.html#mmdet.engine.hooks.DetVisualizationHook
    logger=dict(type='LoggerHook', interval=interval))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

vis_backends = [dict(type='TensorboardVisBackend'), dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

