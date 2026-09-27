_base_ = ["../_base_/default_runtime.py"]

# ============================================================
# Runtime
# ============================================================

# 3 GPU DDP
# Pointcept 这里 batch_size 表示总 batch size
# 3 / 3 GPU = 每卡 1 个样本
batch_size = 3
batch_size_val = 3
num_worker = 3
sync_bn = True

epoch = 100
eval_epoch = 100

mix_prob = 0.8
empty_cache = False
enable_amp = False
wandb = False
weight = None
resume = False

# ============================================================
# Dataset
# ============================================================

data_root = "/media/fln/WD1/chenq/datasets/gridnethd_half"

num_classes = 11
ignore_index = 255

class_names = [
    "pylon",
    "conductor",
    "structural_cable",
    "insulator",
    "high_vegetation",
    "low_vegetation",
    "herbaceous",
    "ground",
    "road",
    "water",
    "building",
]

# RTX 4080 16GB：
# 原始 block 最大约 175 万点，因此限制单样本最大点数
point_max = 50000

# ============================================================
# Model
# ============================================================

model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,

    backbone=dict(
        type="PT-v3m1",
        in_channels=6,

        order=(
            "z",
            "z-trans",
            "hilbert",
            "hilbert-trans",
        ),

        stride=(2, 2, 2, 2),

        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),

        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),

        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,

        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,

        shuffle_orders=True,
        pre_norm=True,

        enable_rpe=False,
        enable_flash=False,

        upcast_attention=False,
        upcast_softmax=False,
    ),

    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
)

# ============================================================
# Optimizer
# ============================================================

optimizer = dict(
    type="AdamW",
    lr=0.002,
    weight_decay=0.005,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

param_dicts = [
    dict(
        keyword="block",
        lr=0.0002,
    )
]

# ============================================================
# Hooks
# ============================================================

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),

    dict(
        type="SemSegEvaluator",
        write_cls_metrics=True,
    ),

    dict(
        type="TrainingMetricsHook",
        output_dir="training_process",
    ),

    dict(
        type="CheckpointSaver",
        save_freq=None,
    ),
]

# ============================================================
# Data
# ============================================================

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    train=dict(
        type="Gridnethd",
        split="train_final",
        data_root=data_root,

        transform=[
            dict(type="CenterShift", apply_z=True),

            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),

            dict(
                type="RandomScale",
                scale=[0.9, 1.1],
            ),

            dict(
                type="RandomFlip",
                p=0.5,
            ),

            dict(
                type="RandomJitter",
                sigma=0.005,
                clip=0.02,
            ),

            dict(
                type="ChromaticAutoContrast",
                p=0.2,
                blend_factor=None,
            ),

            dict(
                type="ChromaticTranslation",
                p=0.95,
                ratio=0.05,
            ),

            dict(
                type="ChromaticJitter",
                p=0.95,
                std=0.05,
            ),

            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),

            dict(
                type="SphereCrop",
                point_max=point_max,
                mode="random",
            ),

            dict(type="CenterShift", apply_z=False),

            dict(type="NormalizeColor"),

            dict(type="ToTensor"),

            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                ),
                feat_keys=(
                    "color",
                    "coord",
                ),
            ),
        ],

        test_mode=False,
    ),

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    val=dict(
        type="Gridnethd",
        split="val_final",
        data_root=data_root,

        transform=[
            dict(type="CenterShift", apply_z=True),

            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),

            dict(type="CenterShift", apply_z=False),


            dict(type="NormalizeColor"),

            dict(type="ToTensor"),

            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                ),
                feat_keys=(
                    "color",
                    "coord",
                ),
            ),
        ],

        test_mode=False,
    ),
)

# ============================================================
# Output
# ============================================================

save_path = "exp/gridnethd/ptv3_half_centershift_100ep"
enable_wandb = False
