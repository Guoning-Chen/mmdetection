_base_ = ['../_base_/models/mask_rcnn_r50_fpn_sk.py',
          '../_base_/datasets/sk_instance.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

model = dict(pretrained='torchvision://resnet101',
             backbone=dict(depth=101))

# //============================== schedules =================================//
# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=86,
    warmup_ratio=0.02,
    step=[6, 11])  # start from 0
total_epochs = 12

# //============================== runtime ===================================//
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
load_from = 'checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

