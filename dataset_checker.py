import os
import random

import cv2 as cv
import numpy as np
import torch
from kornia.geometry.conversions import (axis_angle_to_quaternion,
                                         quaternion_to_rotation_matrix)
from kornia.geometry.epipolar import essential_from_Rt

from utils.vision.opencv.drawing_utils import draw_epipolar_line
from utils.vision.opencv.epipolar_geometry import get_F_from_K_E


def _read_image_pair_and_K_q_t(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/0.jpg"
    img1_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/1.jpg"
    K_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/K.txt"
    q_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/q.txt"
    t_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/t.txt"

    assert os.path.exists(img0_path), f"{img0_path} does not exist."
    assert os.path.exists(img1_path), f"{img1_path} does not exist."
    assert os.path.exists(K_path), f"{K_path} does not exist."

    img0 = cv.imread(img0_path, cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    K = np.loadtxt(K_path)

    if os.path.exists(q_path):
        q = np.loadtxt(q_path)
    else: 
        q = None
    
    if os.path.exists(t_path):
        t = np.loadtxt(t_path)
    else:
        t = None

    return img0, img1, K, q, t


def _read_image_pair_and_K_E(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0, img1, K, q, t = _read_image_pair_and_K_q_t(dataset, category, img_pair_name, dataset_dir)

    if q is None:  # Unknown
        assert t is None  # Unknown
        E = None
    
    else:
        R2 = quaternion_to_rotation_matrix(axis_angle_to_quaternion(torch.tensor([0.0, 0.0, 0.0]))).reshape(1, 3, 3)
        R1 = quaternion_to_rotation_matrix(torch.tensor(q, dtype=R2.dtype)).reshape(1, 3, 3)

        t2 = torch.zeros(3, dtype=R1.dtype).reshape(1, 3, 1)
        t1 = torch.tensor(t, dtype=R2.dtype).reshape(1, 3, 1)

        # Important: I had to reorder R2 and R1, also t2 and t1. References are R2 and t2.

        E = essential_from_Rt(R1, t1, R2, t2).reshape(3, 3).numpy()
    return img0, img1, K, E


def read_image_pair_and_F(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0, img1, K, E = _read_image_pair_and_K_E(dataset, category, img_pair_name, dataset_dir)
    if E is None:
        F = None
    else:
        F = get_F_from_K_E(K, E)
    return img0, img1, F


def visualize_image_pair(img0, img1, F, point=None, color=(255, 0, 0)):
    assert F is not None
    if point is None:
        # Generate a random point that is not too close to the borders.
        min_x = img0.shape[1] // 4
        max_x = img0.shape[1] * 3 // 4
        min_y = img0.shape[0] // 4
        max_y = img0.shape[0] * 3 // 4
        x = random.randrange(min_x, max_x)
        y = random.randrange(min_y, max_y)
        point = (x, y)

    img0_with_point, img1_with_line = draw_epipolar_line(img0, img1, F, point, color)
    img = np.hstack((img0_with_point, img1_with_line))
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.setWindowProperty('img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('img', img)
    cv.waitKey(0)
