import os
from pathlib import Path
import shutil
import numpy as np
from regex import P

import torch
from kornia.geometry.conversions import (matrix4x4_to_Rt,
                                         rotation_matrix_to_quaternion)
from kornia.geometry.epipolar import relative_camera_motion


def create_tanks_and_temples_datasets(n=10, input_dir='sources/tanks-and-temples', output_main_dir='datasets/tanks-and-temples'):
    
    input_dir = Path(input_dir)
    main_output_dir = Path(output_main_dir)

    def read_matrix4x4(log_path: str, img_no: int) -> torch.Tensor:
        # Important: Image numbers start from 1. But image indices start from 0.
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
    
    def get_relative_camera_motion_from_matrix4x4(matrix1: torch.Tensor, matrix2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def extract_pose(img1_no: int, img2_no: int, scene: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert img1_no >= 1
        assert img2_no >= 1
        assert img1_no != img2_no
        
        matrix1 = read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img1_no)
        matrix2 = read_matrix4x4(input_dir / f'{scene}_COLMAP_SfM.log', img2_no)
        R, t = get_relative_camera_motion_from_matrix4x4(matrix1, matrix2)

        q = rotation_matrix_to_quaternion(R)

        t = t.reshape(3)
        q = q.reshape(4)

        return q, t
    
    scenes = [scene for scene in os.listdir(input_dir) if os.path.isdir(input_dir / scene)]
    for scene in scenes:
        img_count = len(os.listdir(input_dir / scene))

        # Every n-th image will be used as the first image.
        # The second image will be the previous (n-1) images and the next (n-1) images.

        for img0_no in range(n, img_count - n + 2, n):
            for img1_no in list(range(img0_no - n + 1, img0_no)) + list(range(img0_no + 1, img0_no + n)):
                img1_path = input_dir / scene / f'{img0_no:06}.jpg'
                img2_path = input_dir / scene / f'{img1_no:06}.jpg'

                assert os.path.exists(img1_path)
                assert os.path.exists(img2_path)
                
                output_dir = main_output_dir / scene / str(img0_no) / f'{img0_no}-{img1_no}'
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)

                shutil.copy2(img1_path, output_dir / '0.jpg')
                shutil.copy2(img2_path, output_dir / '1.jpg')

                q, t = extract_pose(img0_no, img1_no, scene)
                np.savetxt(output_dir / 'q.txt', q.numpy())
                np.savetxt(output_dir / 't.txt', t.numpy())

                k_path = input_dir / 'K.txt'
                assert os.path.exists(k_path)
                shutil.copy2(k_path, output_dir / 'K.txt')


if __name__ == '__main__':
    create_tanks_and_temples_datasets()
