"""
Common utility functions.
Stub implementation - needs full implementation from OpenPCDet.
"""

import torch
import numpy as np


def rotate_points_along_z(points, angle):
    """
    Rotate points along z-axis.
    
    Args:
        points: (B, N, 3 + C) points to rotate
        angle: (B,) rotation angles
        
    Returns:
        Rotated points
    """
    # Stub implementation
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    
    if points.shape[-1] == 2:
        # 2D points
        points_rot = points.clone()
        points_rot[..., 0] = cosa * points[..., 0] - sina * points[..., 1]
        points_rot[..., 1] = sina * points[..., 0] + cosa * points[..., 1]
        return points_rot
    else:
        # 3D or higher dimensional points
        points_rot = points.clone()
        points_rot[..., 0] = cosa.unsqueeze(-1) * points[..., 0] - sina.unsqueeze(-1) * points[..., 1]
        points_rot[..., 1] = sina.unsqueeze(-1) * points[..., 0] + cosa.unsqueeze(-1) * points[..., 1]
        return points_rot


def limit_period(val, offset=0.5, period=np.pi):
    """
    Limit angles to a specific period.
    
    Args:
        val: Angle values
        offset: Offset for limiting
        period: Period (default: pi)
        
    Returns:
        Limited angles
    """
    if isinstance(val, torch.Tensor):
        return val - torch.floor(val / period + offset) * period
    else:
        return val - np.floor(val / period + offset) * period


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Get voxel centers from voxel coordinates.
    
    Args:
        voxel_coords: (N, 3) voxel coordinates
        downsample_times: Downsample factor
        voxel_size: Voxel size (3,)
        point_cloud_range: Point cloud range (6,)
        
    Returns:
        Voxel centers
    """
    # Stub implementation
    assert voxel_coords.shape[-1] == 3
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=voxel_coords.device)
    pc_range = torch.tensor(point_cloud_range, dtype=torch.float32, device=voxel_coords.device)
    
    voxel_centers = (voxel_coords[:, [2, 1, 0]].float() + 0.5) * voxel_size * downsample_times
    voxel_centers += pc_range[:3]
    
    return voxel_centers
