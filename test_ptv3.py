import torch

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)

device = "cuda"

# 构造一个最小测试点云
num_points = 4096

coord = torch.rand(num_points, 3, device=device) * 10.0
grid_coord = torch.floor(coord / 0.05).int()

# 先用 xyz + rgb 形式的 6 维输入特征模拟
feat = torch.randn(num_points, 6, device=device)

# 单个点云 batch
offset = torch.tensor([num_points], dtype=torch.long, device=device)

data_dict = {
    "coord": coord,
    "grid_coord": grid_coord,
    "feat": feat,
    "offset": offset,
}

model = PointTransformerV3(
    in_channels=6,
    enc_patch_size=(128, 128, 128, 128, 128),
    dec_patch_size=(128, 128, 128, 128),
    enable_flash=False,
).to(device)

model.eval()

with torch.no_grad():
    output = model(data_dict)

print("PTv3 forward OK")
print("output feature shape:", output.feat.shape)
print("device:", output.feat.device)

