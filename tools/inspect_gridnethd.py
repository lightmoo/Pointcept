from pathlib import Path
from PIL import Image
import numpy as np
import laspy

# 当前只检查 t1z5b
root = Path("/mnt/f/Dataset/GridNet-HD/t1z5b")

image_dir = root / "images"
mask_dir = root / "masks"
lidar_file = root / "lidar" / "t1z5b.las"
pose_file = root / "pose" / "cameras_pose.txt"


# --------------------------------------------------
# 1. 检查影像和 mask 数量
# --------------------------------------------------
images = sorted(image_dir.glob("*"))
masks = sorted(mask_dir.glob("*"))

print("=" * 60)
print("1. Image / Mask")
print("=" * 60)

print("image count:", len(images))
print("mask count :", len(masks))

print("\nfirst image:", images[0].name)
print("first mask :", masks[0].name)


# --------------------------------------------------
# 2. 检查影像尺寸
# --------------------------------------------------
img = Image.open(images[0])

print("\nimage mode :", img.mode)
print("image size :", img.size)


# --------------------------------------------------
# 3. 检查 mask
# --------------------------------------------------
mask = Image.open(masks[0])
mask_np = np.array(mask)

print("\nmask mode  :", mask.mode)
print("mask size  :", mask.size)
print("mask shape :", mask_np.shape)
print("mask dtype :", mask_np.dtype)

if mask_np.ndim == 2:
    unique_values = np.unique(mask_np)
    print("unique mask labels:")
    print(unique_values[:100])
    print("label count:", len(unique_values))

elif mask_np.ndim == 3:
    colors = np.unique(
        mask_np.reshape(-1, mask_np.shape[-1]),
        axis=0
    )

    print("unique mask colors:")
    print(colors[:100])
    print("color count:", len(colors))


# --------------------------------------------------
# 4. 检查 LAS 点云
# --------------------------------------------------
print("\n" + "=" * 60)
print("2. LiDAR")
print("=" * 60)

las = laspy.read(lidar_file)

print("point count :", len(las.points))
print("point format:", las.header.point_format)
print("dimensions  :", list(las.point_format.dimension_names))

xyz = np.column_stack((las.x, las.y, las.z))

print("\nXYZ shape:", xyz.shape)

print("XYZ min:")
print(xyz.min(axis=0))

print("XYZ max:")
print(xyz.max(axis=0))


# --------------------------------------------------
# 5. 检查 LAS 是否存在 classification
# --------------------------------------------------
dims = list(las.point_format.dimension_names)

if "classification" in dims:
    classes, counts = np.unique(
        np.asarray(las.classification),
        return_counts=True
    )

    print("\nLAS classification:")
    for cls, count in zip(classes, counts):
        print(f"class {cls}: {count}")
else:
    print("\nNo classification field found in LAS.")

# --------------------------------------------------
# 检查 GridNet-HD 自带的 ground_truth 标签
# --------------------------------------------------
if "ground_truth" in dims:
    gt = np.asarray(las["ground_truth"])

    classes, counts = np.unique(gt, return_counts=True)

    print("\nGridNet-HD ground_truth:")
    print("dtype:", gt.dtype)
    print("shape:", gt.shape)
    print("unique labels:", classes)
    print("class count:", len(classes))

    print("\nLabel statistics:")
    for cls, count in zip(classes, counts):
        ratio = count / len(gt) * 100
        print(f"class {cls:3}: {count:10} points ({ratio:.4f}%)")
else:
    print("\nNo ground_truth field found.")

# --------------------------------------------------
# 6. 检查 pose
# --------------------------------------------------
print("\n" + "=" * 60)
print("3. Camera Pose")
print("=" * 60)

with open(pose_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("pose line count:", len(lines))

print("\nfirst 5 pose lines:")
for line in lines[:5]:
    print(line.strip())