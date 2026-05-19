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

    def get_test_case(self, query_size=1, gallery_size=10, mode="random", max_positives=1):
        """
        全功能终极版 DataLoader：
        1. 支持 query_size > 1：返回一个包含多张【相同身份、不同视角】图片的列表 (List[str])。
        2. 支持在一个 Gallery 中随机混入 1 到 max_positives 个相同的目标人物。
        3. 返回的 ground_truth_indices 是一个包含所有正样本位置的列表 (List[int])。
        """
        # =================【核心修改区：多 Query 抽取】=================
        # 1. 先盲抽一个基础 Query，用来确定这次要找的 PID
        base_query_path = random.choice(self.query_paths)
        target_pid, _ = self._parse_filename(base_query_path)
        
        # 2. 从所有 Query 库中，把属于这个 PID 的所有图片全捞出来
        query_pool = []
        for p in self.query_paths:
            pid, camid = self._parse_filename(p)
            if pid == target_pid:
                query_pool.append((p, camid))
                
        # 3. 尽量选择不同摄像头的图片作为 Query（信息互补原则）
        # 先按摄像头去重分组，优先保证视角多样性
        unique_cam_queries = {}
        for p, cam in query_pool:
            if cam not in unique_cam_queries:
                unique_cam_queries[cam] = p
                
        # 提取出了多样化视角的 Query
        diverse_queries = list(unique_cam_queries.values())
        
        # 如果多样化视角不够 query_size，就从纯图片池里不看摄像头地补齐
        if len(diverse_queries) >= query_size:
            final_query_paths = random.sample(diverse_queries, query_size)
        else:
            # 视角不够用，就拿全部属于该 PID 的图片随机补，但上限不能超过图片库总数
            all_pid_images = [p for p, cam in query_pool]
            actual_q_size = min(len(all_pid_images), query_size)
            final_query_paths = random.sample(all_pid_images, actual_q_size)
        # =============================================================
        
        # 4. 划分 Gallery 的正负样本池
        gallery_full = [(p, *self._parse_filename(p)) for p in self.gallery_paths]
        # 注意：为了严谨，Gallery 里的图片不能和已经作为 Query 的图片重复
        positives = [p for (p, pid, cam) in gallery_full if (pid == target_pid and p not in final_query_paths)]
        negatives = [p for (p, pid, cam) in gallery_full if pid != target_pid]
        
        # 容错：万一这个 ID 实在没有其他 Gallery 样本了，重新抽题
        if not positives:
            return self.get_test_case(query_size, gallery_size, mode, max_positives)
            
        # 5. 决定这次放入几个正确的候选人
        num_pos = random.randint(1, min(len(positives), max_positives, gallery_size))
        
        # 抽取多个正样本
        target_positives = random.sample(positives, num_pos)
        
        # 剩下的坑位用负样本填满
        num_negatives = gallery_size - num_pos
        curr_negatives = random.sample(negatives, min(len(negatives), num_negatives))
        
        # 合并并打乱
        gallery_subset = target_positives + curr_negatives
        random.shuffle(gallery_subset)
        
        # 找出所有正样本在打乱后的索引位置
        gt_indices = [i for i, path in enumerate(gallery_subset) if path in target_positives]
        
        # 返回结果：final_query_paths 变成了一个路径列表！
        return final_query_paths, gallery_subset, gt_indices