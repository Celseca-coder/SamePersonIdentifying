import os
import random
import glob
import re
from PIL import Image

class Market1501Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Check paths - using raw strings or forward slashes to avoid escape issues
        self.query_dir = os.path.join(data_dir, "query")
        # Handle potential multiple directory structures for gallery
        test_dir = os.path.join(data_dir, "bounding_box_test")
        if not os.path.exists(test_dir):
             # Fallback if structure is slightly different
             test_dir = os.path.join(data_dir, "gt_bbox") # Or wherever gallery is
        
        self.gallery_dir = test_dir
        
        self.query_paths = self._get_image_paths(self.query_dir)
        self.gallery_paths = self._get_image_paths(self.gallery_dir)
        
        print(f"Loaded {len(self.query_paths)} query images and {len(self.gallery_paths)} gallery images.")

    def _get_image_paths(self, directory):
        if not os.path.exists(directory):
            return []
        # Return logical paths
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")]

    def _parse_filename(self, filename):
        # 0001_c1s1_001051_00.jpg
        # returns pid, camid
        basename = os.path.basename(filename)
        if not basename.endswith(".jpg"):
            return -1, -1
        parts = basename.split("_")
        pid = int(parts[0])
        camid = parts[1]
        return pid, camid

    def get_test_case(self, gallery_size=10, mode="random", max_positives=5):
        """
        升级版 DataLoader：
        支持在一个 Gallery 中随机混入 1 到 max_positives 个相同的目标人物。
        返回的 ground_truth_indices 现在是一个列表 (List[int])！
        """
        # 1. 挑选 Query
        query_path = random.choice(self.query_paths)
        q_pid, q_camid = self._parse_filename(query_path)
        
        # 2. 划分正负样本池
        gallery_full = [(p, *self._parse_filename(p)) for p in self.gallery_paths]
        positives = [p for (p, pid, cam) in gallery_full if pid == q_pid]
        negatives = [p for (p, pid, cam) in gallery_full if pid != q_pid]
        
        if not positives:
            return self.get_test_case(gallery_size, mode)
            
        # =================【核心修改区】=================
        # 决定这次放入几个正确的候选人（例如随机放 1 到 3 个，但不能超过图库总大小）
        num_pos = random.randint(1, min(len(positives), max_positives, gallery_size))
        
        # 抽取多个正样本
        target_positives = random.sample(positives, num_pos)
        
        # 剩下的坑位用负样本填满
        num_negatives = gallery_size - num_pos
        curr_negatives = random.sample(negatives, min(len(negatives), num_negatives))
        
        # 合并并打乱
        gallery_subset = target_positives + curr_negatives
        random.shuffle(gallery_subset)
        
        # 找出所有正样本在打乱后的索引位置，返回一个列表！
        gt_indices = [i for i, path in enumerate(gallery_subset) if path in target_positives]
        # ===============================================
        
        return query_path, gallery_subset, gt_indices