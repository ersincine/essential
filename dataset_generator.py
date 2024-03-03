import os
import shutil
from pathlib import Path

import numpy as np
import torch
from kornia.geometry.conversions import (matrix4x4_to_Rt,
                                         rotation_matrix_to_quaternion)
from kornia.geometry.epipolar import relative_camera_motion
from regex import R


def angle_between_quaternions(q1: np.ndarray, q2: np.ndarray, eps: float=1e-15) -> float:
    assert isinstance(q1, np.ndarray)
    assert isinstance(q2, np.ndarray)
    assert len(q1.shape) == 1 and q1.shape[0] == 4
    assert len(q2.shape) == 1 and q2.shape[0] == 4
    
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.abs(np.dot(q1, q2)))  # ?
    """
    
    q1 = q1 / (np.linalg.norm(q1) + eps)
    q2 = q2 / (np.linalg.norm(q2) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q1 * q2)**2))
    return np.arccos(1 - 2 * loss_q)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, eps: float=1e-15) -> float:
    # We use this function for translation vectors.
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1
    assert v1.shape == v2.shape
    
    v1 = v1 / (np.linalg.norm(v1) + eps)
    v2 = v2 / (np.linalg.norm(v2) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(v1 * v2)**2))
    return np.arccos(np.sqrt(1 - loss_t))


def _read_matrix4x4(log_path: Path, img_no: int) -> torch.Tensor:
    # Important: Image numbers start from 1. But image indices start from 0.

    # A matrix4x4 is a homogeneous transformation matrix.
    # See http://redwood-data.org/indoor/fileformat.html for Trajectory File (.log) Format.
    assert img_no >= 1
    img_idx = img_no - 1

    with open(log_path, 'r') as f:
        lines = f.read().splitlines()
    
    line_idx = img_idx * 5
    assert lines[line_idx] == f'{img_idx} {img_idx} 0'
    lines = lines[line_idx + 1: line_idx + 5]
    matrix = torch.tensor([list(map(float, line.split())) for line in lines])
    assert matrix.shape == (4, 4)
    return matrix


def _get_relative_camera_motion_from_matrix4x4(matrix1: torch.Tensor, matrix2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert matrix1.shape == (4, 4)
    assert matrix2.shape == (4, 4)

    matrix1 = matrix1.reshape(1, 4, 4)
    matrix2 = matrix2.reshape(1, 4, 4)

    R1, t1 = matrix4x4_to_Rt(matrix1)
    R2, t2 = matrix4x4_to_Rt(matrix2)

    assert R1.shape == (1, 3, 3)
    assert t1.shape == (1, 3, 1)
    assert R2.shape == (1, 3, 3)
    assert t2.shape == (1, 3, 1)

    R, t = relative_camera_motion(R1, t1, R2, t2)
    return R, t


def _extract_pose(input_dir: Path, scene: str, img1_no: int, img2_no: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert img1_no >= 1
    assert img2_no >= 1
    assert img1_no != img2_no
    
    matrix1 = _read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img1_no)
    matrix2 = _read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img2_no)
    R, t = _get_relative_camera_motion_from_matrix4x4(matrix1, matrix2)

    q = rotation_matrix_to_quaternion(R)

    t = t.reshape(3)
    q = q.reshape(4)

    return q, t


def _create_tanks_and_temples_datasets_for_img_pairs(input_dir: Path, main_output_dir: Path, img_pairs: list[tuple[str, int, int]], scene_suffix: str) -> None:
    for scene, img0_no, img1_no in img_pairs:
        img0_path = input_dir / scene / f'{img0_no:06}.jpg'
        img1_path = input_dir / scene / f'{img1_no:06}.jpg'

        assert os.path.exists(img0_path)
        assert os.path.exists(img1_path)
        
        output_dir = main_output_dir / (scene + scene_suffix) / str(img0_no) / f'{img0_no}-{img1_no}'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        shutil.copy2(img0_path, output_dir / '0.jpg')
        shutil.copy2(img1_path, output_dir / '1.jpg')

        q, t = _extract_pose(input_dir, scene, img0_no, img1_no)
        np.savetxt(output_dir / 'q.txt', q.numpy())
        np.savetxt(output_dir / 't.txt', t.numpy())

        k_path = input_dir / 'K.txt'
        assert os.path.exists(k_path)
        shutil.copy2(k_path, output_dir / 'K.txt')


def create_tanks_and_temples_datasets_uniformly(n=10, input_dir='sources/tanks-and-temples', output_main_dir='datasets/tanks-and-temples'):
    input_dir = Path(input_dir)
    main_output_dir = Path(output_main_dir)

    scenes = [scene for scene in os.listdir(input_dir) if os.path.isdir(input_dir / scene)]
    img_pairs = []
    for scene in scenes:
        img_count = len(os.listdir(input_dir / scene))

        # Every n-th image will be used as the first image.
        # The second image will be the previous (n-1) images and the next (n-1) images.

        for img0_no in range(n, img_count - n + 2, n):
            for img1_no in list(range(img0_no - n + 1, img0_no)) + list(range(img0_no + 1, img0_no + n)):
                    img_pairs.append((scene, img0_no, img1_no))

    _create_tanks_and_temples_datasets_for_img_pairs(input_dir, main_output_dir, img_pairs, '')


def create_tanks_and_temples_datasets_with_controls(n=10, min_angle_orientation=5, max_angle_orientation=20,
                                                input_dir='sources/tanks-and-temples', output_main_dir='datasets/tanks-and-temples'):    
    
    # FIXME TODO Translationlarda açı belirleyemiyoruz. Nasıl olur bilmiyorum. Şimdi hep 90 geliyor.
    
    input_dir = Path(input_dir)
    main_output_dir = Path(output_main_dir)

    def conditions_hold(scene, img0_no, img1_no):
        q, _ = _extract_pose(input_dir, scene, img0_no, img1_no)
        q = q.numpy()

        reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (0, 0, 0) as axis-angle 
        angle_orientation = angle_between_quaternions(q, reference_quaternion)
        angle_orientation = np.degrees(angle_orientation)

        return min_angle_orientation <= angle_orientation <= max_angle_orientation

    scenes = [scene for scene in os.listdir(input_dir) if os.path.isdir(input_dir / scene)]
    img_pairs = []
    for scene in scenes:
        img_count = len(os.listdir(input_dir / scene))

        # Every n-th image will be used as the first image.
        # The second image will be the previous (n-1) images and the next (n-1) images.

        for img0_no in range(n, img_count - n + 2, n):
            for img1_no in list(range(img0_no - n + 1, img0_no)) + list(range(img0_no + 1, img0_no + n)):

                if conditions_hold(scene, img0_no, img1_no):
                    img_pairs.append((scene, img0_no, img1_no))

    _create_tanks_and_temples_datasets_for_img_pairs(input_dir, main_output_dir, img_pairs, f' {min_angle_orientation}-{max_angle_orientation}')


if __name__ == '__main__':
    #create_tanks_and_temples_datasets_uniformly(n=10)
    #create_tanks_and_temples_datasets_with_controls(n=10, min_angle_orientation=5, max_angle_orientation=20, min_angle_translation=5, max_angle_translation=20)

    # identity translation
    t1 = np.array([1.0, 0, 0])
    t2 = np.array([0.0, 1.0, 0])
    print(np.degrees(angle_between_vectors(t1, t2)))

