import os
import random
import re

class PKUSketchDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sketch_dir = os.path.join(data_dir, "sketch")
        self.photo_dir = os.path.join(data_dir, "photo")
        
        self.sketch_paths = self._get_image_paths(self.sketch_dir)
        self.photo_paths = self._get_image_paths(self.photo_dir)
        
        # 构建按 ID 索引的字典 {pid: [path1, path2...]}
        self.sketch_dict = self._build_id_dict(self.sketch_paths)
        self.photo_dict = self._build_id_dict(self.photo_paths)
        
        # 找出同时拥有草图和照片的有效 ID
        self.valid_pids = list(set(self.sketch_dict.keys()) & set(self.photo_dict.keys()))
        print(f"✅ PKUSketch 加载完毕: 找到 {len(self.valid_pids)} 个有效身份ID。")

    def _get_image_paths(self, directory):
        if not os.path.exists(directory):
            print(f"⚠️ 警告: 找不到文件夹 {directory}")
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def _parse_pid(self, filepath):
        """
        解析文件命中的身份 ID (PID)。
        兼容 '1.jpg' 和 '1_05_356.jpg' 两种格式。
        """
        basename = os.path.basename(filepath)
        match = re.search(r'^(\d+)', basename)
        if match:
            return int(match.group(1))
        return -1

    def _build_id_dict(self, paths):
        id_dict = {}
        for p in paths:
            pid = self._parse_pid(p)
            if pid != -1:
                if pid not in id_dict:
                    id_dict[pid] = []
                id_dict[pid].append(p)
        return id_dict

    def get_test_case(self, query_size=1, gallery_size=5):
        """
        抽取测试用例：Query 是草图，Gallery 是照片
        """
        # 数据集限制：每个人只有1张草图
        if query_size > 1:
            print("⚠️ 警告: PKUSketch 数据集每个人只有 1 张草图，已强制将 query_size 设为 1")
            query_size = 1

        if len(self.photo_paths) < gallery_size:
            raise ValueError("Gallery 总图片数少于请求的 gallery_size！")

        # 1. 随机选一个目标 ID
        target_pid = random.choice(self.valid_pids)
        
        # 2. 抽取 Query (草图)
        query_paths = [self.sketch_dict[target_pid][0]] # 直接取那唯一的一张草图
        
        # 3. 抽取 Gallery (照片)
        gallery_subset = []
        
        # 放入1个正样本 (True Positive 照片，随机二选一)
        pos_img = random.choice(self.photo_dict[target_pid])
        gallery_subset.append(pos_img)
        
        # 放入负样本 (Distractors 照片)
        other_pids = [pid for pid in self.photo_dict.keys() if pid != target_pid]
        random.shuffle(other_pids)
        
        for pid in other_pids:
            if len(gallery_subset) >= gallery_size:
                break
            neg_imgs = self.photo_dict[pid]
            if neg_imgs:
                gallery_subset.append(random.choice(neg_imgs))
                
        # 4. 打乱 Gallery，并记录正样本的真实索引
        random.shuffle(gallery_subset)
        gt_indices = [i for i, path in enumerate(gallery_subset) if self._parse_pid(path) == target_pid]
        
        return query_paths, gallery_subset, gt_indices