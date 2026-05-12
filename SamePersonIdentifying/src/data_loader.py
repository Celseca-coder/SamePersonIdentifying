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
        print(f"using {test_dir} as test set / gallery.")
        
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

    def get_test_case(self, gallery_size=10, mode="random"):
        """
        Returns a single test case:
        - query_path: str
        - gallery_paths: List[str] (contains at least one match, and at least one mismatch)
        - gt_indices: List[int] (indices of all the matches in gallery_paths)
        """
        if gallery_size < 2:
            raise ValueError("gallery_size 至少需要为 2，以保证既有正样本也有负样本。")

        # 1. Pick a random query
        query_path = random.choice(self.query_paths)
        q_pid, q_camid = self._parse_filename(query_path)
        
        # 2. Find positives in gallery
        gallery_full = [(p, *self._parse_filename(p)) for p in self.gallery_paths]
        
        positives = [p for (p, pid, cam) in gallery_full if pid == q_pid]
        negatives = [p for (p, pid, cam) in gallery_full if pid != q_pid]
        
        # 如果根本没有正样本，重新抽题
        if not positives:
            return self.get_test_case(gallery_size, mode)
            
        # 3. Sample
        # 计算最多能放多少个正样本（不能全为正样本，所以最多 gallery_size - 1 个；同时不能超过正样本的实际总数）
        max_positives = min(len(positives), gallery_size - 1)
        
        # 随机决定本次抽取几个正样本
        num_positives = random.randint(1, max_positives)
        sampled_positives = random.sample(positives, num_positives)
        
        # 剩下的位置全部用负样本填满
        num_negatives = gallery_size - num_positives
        sampled_negatives = random.sample(negatives, min(len(negatives), num_negatives))
        
        # Combine 组合在一起
        gallery_subset = sampled_positives + sampled_negatives
        random.shuffle(gallery_subset) # 打乱顺序
        
        # Find indices: 找出所有正样本在打乱后列表中的索引
        # 这里用集合(set)加快判断速度
        positives_set = set(sampled_positives)
        gt_indices = [i for i, path in enumerate(gallery_subset) if path in positives_set]
        
        return query_path, gallery_subset, gt_indices