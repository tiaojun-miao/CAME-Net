# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import matplotlib
from came_net import CAMENet
from pga_algebra import create_point_pga, extract_point_coordinates, random_motor

torch.manual_seed(0)

# 使用支持中文的字体，避免 matplotlib 在 PyCharm backend 下提示缺字
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 生成一个简单点云（球面 256 点）
def sample_sphere(num_points=256):
    phi = torch.randn(num_points).acos()  # 随机极角
    theta = torch.rand(num_points) * 2 * torch.pi
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    pts = torch.stack([x, y, z], dim=-1)
    return pts.unsqueeze(0)  # (1, N, 3)


points = sample_sphere(256)

# 2. 构建模型
device = torch.device('cpu')
model = CAMENet(num_classes=10, num_layers=4, num_heads=8).to(device)
model.eval()


def equiv_error(sigma_rot=0.3, sigma_trans=0.3):
    motor = random_motor(1, sigma_rot=sigma_rot, sigma_trans=sigma_trans)
    pts_mv = create_point_pga(points)
    transformed_mv = pts_mv.apply_motor(motor)
    transformed_pts = extract_point_coordinates(transformed_mv)

    with torch.no_grad():
        latent_orig = model.get_latent_multivector(points)
        latent_trans_input = model.get_latent_multivector(transformed_pts)
        latent_trans_output = latent_orig.apply_motor(motor)

    return torch.mean((latent_trans_input.data - latent_trans_output.data) ** 2).item()


sigmas = torch.linspace(0, 1.0, 10)
errors = [equiv_error(sigma, sigma) for sigma in sigmas]

plt.figure(figsize=(6, 4))
plt.plot(sigmas.numpy(), errors, marker='o')
plt.xlabel("Motor σ (旋转 = 平移)")
plt.ylabel("Equivariance MSE")
plt.title("CAME-Net 等变误差 vs motor 幅度")
plt.grid(True)
plt.show()


# 如果想要直观地看 motor 对点云的搬运效果，可以调用这个辅助函数

def visualize_random_motor(sigma_rot=0.5, sigma_trans=0.5):
    motor = random_motor(1, sigma_rot=sigma_rot, sigma_trans=sigma_trans)
    pts_mv = create_point_pga(points)
    transformed_mv = pts_mv.apply_motor(motor)
    transformed_pts = extract_point_coordinates(transformed_mv)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2], s=5)
    ax1.set_title("原始点云")
    ax2.scatter(transformed_pts[0, :, 0], transformed_pts[0, :, 1], transformed_pts[0, :, 2], s=5, c='orange')
    ax2.set_title("motor 作用后")
    plt.tight_layout()
    plt.show()


visualize_random_motor()
