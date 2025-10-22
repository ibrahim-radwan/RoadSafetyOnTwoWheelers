"""
Oriented IoU Loss for rotated bounding boxes
"""

import torch
import numpy as np


def cal_iou(box1, box2):
    """
    Calculate IoU between rotated boxes.
    
    Args:
        box1: Ground truth boxes, shape (N, 5) - [x, y, w, h, theta]
        box2: Predicted boxes, shape (M, 5) - [x, y, w, h, theta]
    
    Returns:
        iou: IoU matrix of shape (N, M)
        u: Union areas
        inter: Intersection areas  
        g_inter: Intersection areas (duplicate for compatibility)
    """
    try:
        # Try using torchvision's box_iou if available for axis-aligned boxes
        # For rotated boxes, we need a more complex calculation
        from torchvision.ops import box_iou
        
        # Convert rotated boxes to axis-aligned for approximation
        # This is a simplified version - proper rotated IoU would use polygon intersection
        box1_aa = convert_to_axis_aligned(box1)
        box2_aa = convert_to_axis_aligned(box2)
        
        iou = box_iou(box1_aa, box2_aa)
        
        # Calculate areas
        area1 = box1[:, 2] * box1[:, 3]
        area2 = box2[:, 2] * box2[:, 3]
        
        # Calculate union and intersection
        area1_expanded = area1.unsqueeze(1).expand_as(iou)
        area2_expanded = area2.unsqueeze(0).expand_as(iou)
        
        inter = iou * (area1_expanded + area2_expanded) / (1 + iou)
        u = area1_expanded + area2_expanded - inter
        
        return iou, u, inter, inter
    except ImportError:
        # Fallback implementation if torchvision is not available
        return calculate_rotated_iou_fallback(box1, box2)


def convert_to_axis_aligned(boxes):
    """
    Convert rotated boxes to axis-aligned bounding boxes.
    
    Args:
        boxes: Tensor of shape (N, 5) - [x, y, w, h, theta]
    
    Returns:
        aa_boxes: Tensor of shape (N, 4) - [x1, y1, x2, y2]
    """
    if boxes.shape[1] < 5:
        # Already axis-aligned or wrong format
        if boxes.shape[1] == 4:
            return boxes
        else:
            raise ValueError(f"Expected boxes with 4 or 5 dimensions, got {boxes.shape[1]}")
    
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    theta = boxes[:, 4] if boxes.shape[1] > 4 else torch.zeros_like(x_center)
    
    # Calculate corner points of rotated rectangle
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Half dimensions
    hw = width / 2
    hh = height / 2
    
    # Corner offsets (before rotation)
    corners_x = torch.stack([-hw, hw, hw, -hw], dim=1)
    corners_y = torch.stack([-hh, -hh, hh, hh], dim=1)
    
    # Rotate corners
    rotated_x = corners_x * cos_theta.unsqueeze(1) - corners_y * sin_theta.unsqueeze(1)
    rotated_y = corners_x * sin_theta.unsqueeze(1) + corners_y * cos_theta.unsqueeze(1)
    
    # Translate to center
    rotated_x = rotated_x + x_center.unsqueeze(1)
    rotated_y = rotated_y + y_center.unsqueeze(1)
    
    # Get axis-aligned bounding box
    x1 = rotated_x.min(dim=1)[0]
    y1 = rotated_y.min(dim=1)[0]
    x2 = rotated_x.max(dim=1)[0]
    y2 = rotated_y.max(dim=1)[0]
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def calculate_rotated_iou_fallback(box1, box2):
    """
    Fallback implementation for rotated IoU calculation.
    Uses axis-aligned approximation.
    
    Args:
        box1: Ground truth boxes, shape (N, 5)
        box2: Predicted boxes, shape (M, 5)
    
    Returns:
        iou, u, inter, g_inter
    """
    # Convert to axis-aligned
    box1_aa = convert_to_axis_aligned(box1)
    box2_aa = convert_to_axis_aligned(box2)
    
    # Calculate areas
    area1 = (box1_aa[:, 2] - box1_aa[:, 0]) * (box1_aa[:, 3] - box1_aa[:, 1])
    area2 = (box2_aa[:, 2] - box2_aa[:, 0]) * (box2_aa[:, 3] - box2_aa[:, 1])
    
    # Broadcast for pairwise comparison
    x1_max = torch.max(box1_aa[:, 0].unsqueeze(1), box2_aa[:, 0].unsqueeze(0))
    y1_max = torch.max(box1_aa[:, 1].unsqueeze(1), box2_aa[:, 1].unsqueeze(0))
    x2_min = torch.min(box1_aa[:, 2].unsqueeze(1), box2_aa[:, 2].unsqueeze(0))
    y2_min = torch.min(box1_aa[:, 3].unsqueeze(1), box2_aa[:, 3].unsqueeze(0))
    
    # Calculate intersection
    inter_w = torch.clamp(x2_min - x1_max, min=0)
    inter_h = torch.clamp(y2_min - y1_max, min=0)
    inter = inter_w * inter_h
    
    # Calculate union
    area1_expanded = area1.unsqueeze(1)
    area2_expanded = area2.unsqueeze(0)
    u = area1_expanded + area2_expanded - inter
    
    # Calculate IoU
    iou = inter / torch.clamp(u, min=1e-6)
    
    return iou, u, inter, inter
