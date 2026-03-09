import numpy as np
import torch

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def fov_sample2ray(fovx, fovy, interval):
    theta_arr = torch.arange(interval / 2, fovx, interval)
    theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
    phi_arr = torch.arange(interval / 2, fovy, interval)
    phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

    return theta_arr.float(), phi_arr.float()

def omni_map_z(m, z, xi=0.0): #1.1
    return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)

def omni_tan(Ks, width, height, step, fov_mod=1, data_device="cuda"):
    # Ks [..., C, 3, 3]
    fx = Ks[..., 0, 0].to(data_device)
    fy = Ks[..., 1, 1].to(data_device)

    # get largest camera fov (capped at np.pi/2)
    FoVx = min(focal2halffov2(fx, width).max().item() * fov_mod, np.pi / 2)
    FoVy = min(focal2halffov2(fy, height).max().item() * fov_mod, np.pi / 2)

    arr_theta, arr_phi = fov_sample2ray(FoVx/2, FoVy/2, step)

    cos_theta = torch.cos(arr_theta)
    cos_phi = torch.cos(arr_phi)

    cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(data_device)
    cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(data_device)

    tan_theta = torch.tan(arr_theta).to(data_device)
    tan_phi = torch.tan(arr_phi).to(data_device)

    omni_tan_theta = omni_map_z(tan_theta, cos_theta)
    omni_tan_phi = omni_map_z(tan_phi, cos_phi)

    tanfovx = np.tan(FoVx)
    tanfovy = np.tan(FoVy)

    return omni_tan_theta, omni_tan_phi, tanfovx, tanfovy