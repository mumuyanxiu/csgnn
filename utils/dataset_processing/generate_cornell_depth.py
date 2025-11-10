import glob
import os
import sys
import numpy as np
from imageio import imsave
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset_processing.image import DepthImage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = parser.parse_args()

    print("=" * 70)
    print("生成 Cornell 深度图")
    print("=" * 70)
    print(f"数据集路径: {args.path}")
    
    # 查找所有点云文件
    pcds = glob.glob(os.path.join(args.path, '*', 'pcd*[0-9].txt'))
    pcds.sort()
    
    total = len(pcds)
    print(f"找到 {total} 个点云文件\n")
    
    if total == 0:
        print("错误: 未找到点云文件！")
        print("请检查路径是否正确，应包含子目录（如 data-1/, data-2/, origin/）")
        sys.exit(1)
    
    # 生成深度图
    for i, pcd in enumerate(pcds, 1):
        try:
            di = DepthImage.from_pcd(pcd, (480, 640))
            di.inpaint()
            
            of_name = pcd.replace('.txt', 'd.tiff')
            imsave(of_name, di.img.astype(np.float32))
            
            if i % 50 == 0 or i == total:
                print(f"进度: {i}/{total} ({i*100//total}%) - {os.path.basename(of_name)}")
        
        except Exception as e:
            print(f"错误 [{i}/{total}]: {pcd} - {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"完成！共生成 {total} 个深度图")
    print("=" * 70)
    print("\n现在可以开始训练:")
    print('python train_ggcnn.py --network hybrid --dataset cornell --dataset-path "C:\\Users\\24806\\Desktop\\cornell" --use-depth 1 --use-rgb 1 --batch-size 8 --epochs 50')
    print("=" * 70)