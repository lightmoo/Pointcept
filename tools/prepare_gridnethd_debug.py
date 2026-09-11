from pathlib import Path
import numpy as np
import laspy


LAS_PATH = Path(
    "/mnt/f/Dataset/GridNet-HD/t1z5b/lidar/t1z5b.las"
)

OUT_ROOT = Path(
    "data/gridnethd_debug"
)

BLOCK_SIZE = 20.0

# 先只保存少量 block，方便 RTX5080 本地调试
MAX_BLOCKS = 12


# 原始标签 -> GridNet-HD 官方训练标签
LABEL_MAP = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0,       # Pylon
    5: 1,                                 # Conductor
    6: 2, 7: 2,                           # Structural cable
    8: 3, 9: 3, 10: 3, 11: 3,            # Insulator
    14: 4,                                # High vegetation
    15: 5,                                # Low vegetation
    16: 6,                                # Herbaceous vegetation
    17: 7, 18: 7,                         # Rock/gravel/soil
    19: 8,                                # Road
    20: 9,                                # Water
    21: 10,                               # Building
    12: 255, 13: 255, 255: 255
}


def remap_labels(labels):
    out = np.full(labels.shape, 255, dtype=np.int64)

    for old_id, new_id in LABEL_MAP.items():
        out[labels == old_id] = new_id

    return out


print("Reading LAS ...")

las = laspy.read(LAS_PATH)

coord = np.column_stack(
    (las.x, las.y, las.z)
).astype(np.float32)

color = np.column_stack(
    (las.red, las.green, las.blue)
)

# LAS RGB通常为16 bit，转成8 bit
if color.max() > 255:
    color = (color >> 8)

color = color.astype(np.uint8)

segment_raw = np.asarray(
    las["ground_truth"],
    dtype=np.int64
)

segment = remap_labels(segment_raw)

print("points:", len(coord))
print("labels:", np.unique(segment))


# ------------------------------------------------
# 建立二维空间 block
# ------------------------------------------------

x_min = coord[:, 0].min()
y_min = coord[:, 1].min()

block_x = np.floor(
    (coord[:, 0] - x_min) / BLOCK_SIZE
).astype(np.int32)

block_y = np.floor(
    (coord[:, 1] - y_min) / BLOCK_SIZE
).astype(np.int32)

block_id = np.column_stack(
    (block_x, block_y)
)

unique_blocks = np.unique(
    block_id,
    axis=0
)

print("total blocks:", len(unique_blocks))


# ------------------------------------------------
# 优先选择含电力设施的 block
# 0 Pylon
# 1 Conductor
# 2 Structural cable
# 3 Insulator
# ------------------------------------------------

selected = []

for bx, by in unique_blocks:

    mask = (
        (block_x == bx) &
        (block_y == by)
    )

    seg_block = segment[mask]

    electrical_count = np.isin(
        seg_block,
        [0, 1, 2, 3]
    ).sum()

    if electrical_count > 0:
        selected.append(
            (electrical_count, bx, by)
        )


selected.sort(reverse=True)

selected = selected[:MAX_BLOCKS]

print("selected blocks:", len(selected))


# ------------------------------------------------
# 保存 Pointcept 格式
# ------------------------------------------------

for i, (_, bx, by) in enumerate(selected):

    mask = (
        (block_x == bx) &
        (block_y == by)
    )

    coord_block = coord[mask].copy()
    color_block = color[mask]
    segment_block = segment[mask]

    # 局部坐标化
    coord_block -= coord_block.min(
        axis=0,
        keepdims=True
    )

    # 前8块训练，后4块验证
    if i < 8:
        split = "train"
    else:
        split = "val"

    sample_dir = (
        OUT_ROOT /
        split /
        f"t1z5b_block_{bx}_{by}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        sample_dir / "coord.npy",
        coord_block.astype(np.float32)
    )

    np.save(
        sample_dir / "color.npy",
        color_block.astype(np.uint8)
    )

    np.save(
        sample_dir / "segment.npy",
        segment_block.astype(np.int64)
    )

    print(
        split,
        sample_dir.name,
        "points:",
        len(coord_block),
        "labels:",
        np.unique(segment_block)
    )


print("Done.")