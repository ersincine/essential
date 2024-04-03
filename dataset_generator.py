import os
import shutil
from pathlib import Path

import numpy as np
import torch
from kornia.geometry import conversions

from utils.vision.opencv import epipolar_geometry


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


def _extract_pose(input_dir: Path, scene: str, img1_no: int, img2_no: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert img1_no >= 1
    assert img2_no >= 1
    assert img1_no != img2_no
    
    matrix1 = _read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img1_no)
    matrix2 = _read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img2_no)
    R, t = epipolar_geometry.get_relative_camera_motion_from_matrix4x4(matrix1, matrix2)

    q = conversions.rotation_matrix_to_quaternion(R)

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


def create_tanks_and_temples_datasets_uniformly(n=10, selected_scenes=None, input_dir='sources/tanks-and-temples', output_main_dir='datasets/tanks-and-temples'):
    input_dir = Path(input_dir)
    main_output_dir = Path(output_main_dir)

    scenes = [scene for scene in os.listdir(input_dir) if os.path.isdir(input_dir / scene)]
    if selected_scenes is not None:
        scenes = [scene for scene in scenes if scene in selected_scenes]
    img_pairs = []
    for scene in scenes:
        img_count = len(os.listdir(input_dir / scene))

        # Every n-th image will be used as the first image.
        # The second image will be the previous (n-1) images and the next (n-1) images.

        for img0_no in range(n, img_count - n + 2, n):
            for img1_no in list(range(img0_no - n + 1, img0_no)) + list(range(img0_no + 1, img0_no + n)):
                    img_pairs.append((scene, img0_no, img1_no))

    _create_tanks_and_temples_datasets_for_img_pairs(input_dir, main_output_dir, img_pairs, '')


def create_tanks_and_temples_datasets_with_controls(n=10, selected_scenes=None, min_angle_orientation=5, max_angle_orientation=20, min_norm_translation=0.5, max_norm_translation=float('inf'),
                                                input_dir='sources/tanks-and-temples', output_main_dir='datasets/tanks-and-temples'):    
    
    # FIXME TODO Translationlarda açı belirleyemiyoruz. Nasıl olur bilmiyorum. Şimdi hep 90 geliyor.
    
    input_dir = Path(input_dir)
    main_output_dir = Path(output_main_dir)

    def conditions_hold(scene, img0_no, img1_no):
        q, t = _extract_pose(input_dir, scene, img0_no, img1_no)
        q = q.numpy()
        t = t.numpy()

        reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (0, 0, 0) as axis-angle 
        angle_orientation = epipolar_geometry.angle_between_quaternions(q, reference_quaternion)
        angle_orientation = np.degrees(angle_orientation)

        return min_angle_orientation <= angle_orientation <= max_angle_orientation and min_norm_translation <= np.linalg.norm(t) <= max_norm_translation

    scenes = [scene for scene in os.listdir(input_dir) if os.path.isdir(input_dir / scene)]
    if selected_scenes is not None:
        scenes = [scene for scene in scenes if scene in selected_scenes]
    img_pairs = []
    for scene in scenes:
        img_count = len(os.listdir(input_dir / scene))

        # Every n-th image will be used as the first image.
        # The second image will be the previous (n-1) images and the next (n-1) images.

        for img0_no in range(n, img_count - n + 2, n):
            for img1_no in list(range(img0_no - n + 1, img0_no)) + list(range(img0_no + 1, img0_no + n)):

                if conditions_hold(scene, img0_no, img1_no):
                    img_pairs.append((scene, img0_no, img1_no))

    _create_tanks_and_temples_datasets_for_img_pairs(input_dir, main_output_dir, img_pairs, f' {n} {min_angle_orientation}-{max_angle_orientation} {min_norm_translation}-{max_norm_translation}')


def copy_manually_created_datasets(source_dir='sources', destination_dir='datasets'):
    # Copy everything from sources/manually-created-datasets to datasets
    source_dir = Path(source_dir)
    input_dir = source_dir / 'manually-created-datasets'
    main_output_dir = Path(destination_dir)
    # copy dirs inside input_dir
    for dataset in os.listdir(input_dir):
        dataset_path = input_dir / dataset
        if os.path.isdir(dataset_path):
            output_dir = main_output_dir / dataset
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(dataset_path, output_dir)

if __name__ == '__main__':
    #create_tanks_and_temples_datasets_uniformly(n=10)
    #create_tanks_and_temples_datasets_with_controls(n=10, min_angle_orientation=5, max_angle_orientation=20, min_angle_translation=5, max_angle_translation=20)
    pass
