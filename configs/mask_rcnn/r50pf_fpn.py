_base_ = ['../_base_/models/mask_rcnn_r50pf_fpn_sk.py',
          '../_base_/datasets/sk_instance.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

# //============================== model =====================================//
model = dict(
    backbone=dict(
        type='ResNetPf',
        depth=50,
        pf_cfg=[64, 64, 256, 64, 64, 256, 64, 64, 256,
                128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
                25, 25, 1024, 25, 25, 1024, 25, 25, 1024, 25, 25, 1024,
                256, 256, 1024, 25, 25, 1024,
                51, 51, 2048, 51, 51, 2048, 51, 51, 2048])
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
    step=[])  # decay lr after these epochs12, 20
total_epochs = 36

# //============================== runtime ===================================//
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
load_from = 'work_dirs/r50_fpn/r50_prs0099.pth'

