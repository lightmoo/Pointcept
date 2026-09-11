_base_ = ["../_base_/default_runtime.py"]

# ============================================================
# 1. Runtime
# ============================================================

# 本地 RTX 5080 全量数据训练
batch_size = 1
batch_size_val = 1
num_worker = 1

# 全量数据训练 100 个 epoch
epoch = 100
eval_epoch = 100

evaluate = True
test_only = False

mix_prob = 0.0
empty_cache = True

# AMP 混合精度
enable_amp = True

# 不使用 W&B
enable_wandb = False

# 输出目录
save_path = "exp/gridnethd/ptv3_full_100ep"


# ============================================================
# 2. Hooks
# ============================================================
# 重点：
# 当前全量数据只有 train_final / val_final，没有 test。
# 因此去掉默认的 PreciseEvaluator，避免训练结束后自动调用 data.test。

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="SemSegEvaluator"),
    # dict(type="CheckpointSaver", save_freq=None),
        # 先计算验证集指标
    dict(type="SemSegEvaluator", write_cls_metrics=False),

    # 再记录全部过程，并判断当前Epoch是否为Best
    dict(type="TrainingMetricsHook", output_dir="training_process"),

    # 最后由Pointcept保存model_best.pth
    dict(type="CheckpointSaver", save_freq=None),
]


# ============================================================
# 3. Model
# ============================================================

model = dict(
    type="DefaultSegmentorV2",

    # GridNet-HD：11 类语义分割
    num_classes=11,

    # PTv3 decoder 最终输出 64 维点特征
    backbone_out_channels=64,

    backbone=dict(
        type="PT-v3m1",

        # 当前 baseline：
        # feat 只使用 RGB，因此输入特征维数为 3
        # XYZ 单独由 coord / grid_coord 表示
        in_channels=3,

        order=(
            "z",
            "z-trans",
            "hilbert",
            "hilbert-trans",
        ),

        stride=(2, 2, 2, 2),

        # Encoder
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),

        # Decoder
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

        # 暂时不使用 RPE
        enable_rpe=False,

        # 本地 RTX 5080 当前未配置 FlashAttention
        enable_flash=False,

        upcast_attention=False,
        upcast_softmax=False,

        # 当前新版 Pointcept 使用 enc_mode
        enc_mode=False,

        # PDNorm 暂时关闭
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=(
            "ScanNet",
            "S3DIS",
            "Structured3D",
        ),
    ),

    # 语义分割损失
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=255,
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=255,
        ),
    ],
)


# ============================================================
# 4. Optimizer
# ============================================================

optimizer = dict(
    type="AdamW",
    lr=0.002,
    weight_decay=0.005,
)

# PTv3 block 使用更小学习率
param_dicts = [
    dict(
        keyword="block",
        lr=0.0002,
    )
]

scheduler = dict(
    type="OneCycleLR",

    # 普通参数 + block 参数
    max_lr=[0.002, 0.0002],

    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)


# ============================================================
# 5. Dataset
# ============================================================

dataset_type = "Gridnethd"

# GridNet-HD 全量数据
data_root = "/home/chenqiong/datasets/gridnethd_full"

ignore_index = 255


# ============================================================
# 6. Class Names
# ============================================================

names = [
    "pylon",              # 0  杆塔
    "conductor",          # 1  导线
    "structural_cable",   # 2  结构线
    "insulator",          # 3  绝缘子
    "high_vegetation",    # 4
    "low_vegetation",     # 5
    "herbaceous",         # 6
    "ground",             # 7  rock / gravel / soil
    "road",               # 8
    "water",              # 9
    "building",           # 10
]


# ============================================================
# 7. Data Pipeline
# ============================================================

data = dict(
    num_classes=11,
    ignore_index=ignore_index,
    names=names,

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    train=dict(
        type=dataset_type,
        split="train_final",
        data_root=data_root,

        transform=[
            # ---------- 数据增强 ----------

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

            # ---------- 点云网格采样 ----------
            # 生成 PTv3 所需的 grid_coord
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),

            # RGB 归一化
            dict(type="NormalizeColor"),

            # numpy -> torch.Tensor
            dict(type="ToTensor"),

            # 最终交给模型：
            #
            # coord       XYZ
            # grid_coord  体素坐标
            # segment     Ground Truth
            # feat        color
            #
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                ),
                feat_keys=("color",),
            ),
        ],

        test_mode=False,
        ignore_index=ignore_index,
    ),

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    val=dict(
        type=dataset_type,
        split="val_final",
        data_root=data_root,

        transform=[
            # 保存原始 segment
            dict(
                type="Copy",
                keys_dict={
                    "segment": "origin_segment"
                },
            ),

            # Validation 不做随机增强
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),

            dict(type="NormalizeColor"),

            dict(type="ToTensor"),

            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "inverse",
                ),
                feat_keys=("color",),
            ),
        ],

        test_mode=False,
        ignore_index=ignore_index,
    ),
)
