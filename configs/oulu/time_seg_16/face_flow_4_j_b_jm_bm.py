multiflow = dict(
    type='RecognizerGCNMultiflow',
    which_flow=["j", "b", "jm", "bm"],
    flow_weight={"j": 1.0, "b": 1.0, "jm": 0.5, "bm": 0.5},
    num_classes=6,
)
model_j = dict(
    type='RecognizerGCNSingleFlow',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='face68', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=6,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=multiflow["flow_weight"]["j"])),
    softmax=True)

model_b = dict(
    type='RecognizerGCNSingleFlow',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='face68', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=6,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=multiflow["flow_weight"]["b"])),
    softmax=True)

model_jm = dict(
    type='RecognizerGCNSingleFlow',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='face68', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=6,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=multiflow["flow_weight"]["jm"])),
    softmax=True)

model_bm = dict(
    type='RecognizerGCNSingleFlow',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='face68', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=6,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=multiflow["flow_weight"]["bm"])),
    softmax=True)

dataset_type = 'PoseDataset'
ann_file = r"..\data\oulu_10_fold\oulu_0.pkl"
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='face68', feats=["j", "b", "jm", "bm"]),
    dict(type='UniformSample', clip_len=16),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='face68', feats=["j", "b", "jm", "bm"]),
    dict(type='UniformSample', clip_len=16, num_clips=2, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='face68', feats=["j", "b", "jm", "bm"]),
    dict(type='UniformSample', clip_len=16, num_clips=2, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 100
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'

work_dir = r'..\work_dirs\\oulu_10_fold\16_segment\landmarks_0\j_b_jm_bm'
