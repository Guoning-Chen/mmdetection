_base_ = ['../_base_/models/mask_rcnn_r50pf_fpn_sk.py',
          '../_base_/datasets/sk_instance.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

# //============================== model =====================================//
model = dict(
    backbone=dict(
        type='ResNetPf',
        depth=50,
        pf_cfg=None)
)

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
    step=[26])  # decay lr after these epochs12, 20
total_epochs = 48

# //============================== runtime ===================================//
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

