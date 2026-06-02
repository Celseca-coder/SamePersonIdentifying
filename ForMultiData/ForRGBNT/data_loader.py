import os
import random
import re

class RGBNTDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.modality_dirs = {
            "RGB": "RGB",
            "NI": "NI", 
            "TI": "TI"
        }
        
        self.modalities = list(self.modality_dirs.keys())
        # 数据结构: { modality: { pid: [(filepath, camid), ...] } }
        self.data_dict = {mod: {} for mod in self.modalities} 
        self.all_pids = set()
        
        self._build_dataset()
        
        # 找出三种模态都有图片的“公共ID”，保证实验的严谨性和公平性
        self.valid_pids = list(self.all_pids)
        for mod in self.modalities:
            self.valid_pids = [pid for pid in self.valid_pids if pid in self.data_dict[mod]]
            
        print(f"✅ RGBNT201 加载完毕: 找到 {len(self.valid_pids)} 个同时具备 RGB, NIR, TI 三种模态的有效身份ID。")

    def _parse_filename(self, filepath):
        """
        解析文件名前几位数字作为 ID，第二项作为 Camera ID
        示例: 000009_cam3_1_00.jpg -> pid=9, camid='cam3'
        """
        basename = os.path.basename(filepath)
        if not basename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            return -1, "unknown"
            
        parts = basename.split('_')
        if len(parts) >= 2:
            try:
                pid = int(parts[0])  # 提取 000009 为整型 9
                camid = parts[1]     # 提取 cam3
                return pid, camid
            except ValueError:
                return -1, "unknown"
        return -1, "unknown"

    def _build_dataset(self):
        for mod, folder_name in self.modality_dirs.items():
            mod_dir = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(mod_dir):
                print(f"⚠️ 警告: 找不到 {mod} 模态的文件夹: {mod_dir}")
                continue
                
            for file in os.listdir(mod_dir):
                filepath = os.path.join(mod_dir, file)
                pid, camid = self._parse_filename(filepath)
                
                if pid != -1:
                    self.all_pids.add(pid)
                    if pid not in self.data_dict[mod]:
                        self.data_dict[mod][pid] = []
                    # 将图片路径和摄像头ID作为元组存入，方便后续跨摄像头筛选
                    self.data_dict[mod][pid].append((filepath, camid))

    def get_mixed_gallery_test_case(self, query_mod="RGB", query_size=1, gallery_size=5):
        """
        🚀 终极混合图库生成器：
        1. 支持多 Query 图像
        2. 强制 Query 与 Gallery 正样本不重叠 (Overlap Prevention)
        3. 强制跨摄像头检索 (Cross-Camera Retrieval)
        4. Gallery 包含随机混合的光谱模态
        """
        # 1. 筛选出在目标模态下图片数量满足 query_size 要求的有效 ID
        valid_targets = [pid for pid in self.valid_pids if len(self.data_dict[query_mod][pid]) >= query_size]
        if not valid_targets:
            print(f"⚠️ 警告: 没有 ID 在 {query_mod} 模态下满足 query_size={query_size}，将自动随机降级。")
            valid_targets = self.valid_pids
            
        target_pid = random.choice(valid_targets)
        
        # 2. 抽取 Query (可见光 RGB)
        available_queries = self.data_dict[query_mod][target_pid] # List of (path, cam)
        
        # 优先抽取不同摄像头的图片作为多 Query，以提供更丰富的视角信息
        cam_dict = {}
        for p, cam in available_queries:
            if cam not in cam_dict: cam_dict[cam] = []
            cam_dict[cam].append(p)
            
        query_paths = []
        query_cams = set()
        cams_list = list(cam_dict.keys())
        random.shuffle(cams_list)
        
        # 第一轮：每个摄像头尽量抽一张
        for cam in cams_list:
            if len(query_paths) < query_size:
                p = random.choice(cam_dict[cam])
                query_paths.append(p)
                query_cams.add(cam)
                cam_dict[cam].remove(p)
                
        # 第二轮：如果还不够 query_size，无视摄像头继续补齐
        remaining_queries = [p for p, cam in available_queries if p not in query_paths]
        while len(query_paths) < query_size and remaining_queries:
            p = random.choice(remaining_queries)
            query_paths.append(p)
            remaining_queries.remove(p)
            # 记录补齐图片的摄像头
            for ap, acam in available_queries:
                if ap == p:
                    query_cams.add(acam)
                    break

        # 3. 抽取 Gallery 正样本 (混合模态 & 跨摄像头)
        gallery_subset = []
        mods = list(self.modalities)
        random.shuffle(mods) # 打乱模态，随机决定目标的真实模态
        
        pos_img_path = None
        for pos_mod in mods:
            all_pos_candidates = self.data_dict[pos_mod][target_pid]
            
            # 【核心学术限制】: 
            # 条件A: 图片不能和 Query 里的图片是同一张 (防重叠)
            # 条件B: 图片的摄像头不能和 Query 里任意一张图片的摄像头相同 (跨摄像头)
            cross_cam_candidates = [p for p, cam in all_pos_candidates 
                                    if p not in query_paths and cam not in query_cams]
            
            if cross_cam_candidates:
                pos_img_path = random.choice(cross_cam_candidates)
                break
            else:
                # 降级容错：如果没有跨摄像头的图，退一步允许相同摄像头，但【绝对不允许同一张图】
                non_overlap_candidates = [p for p, cam in all_pos_candidates if p not in query_paths]
                if non_overlap_candidates:
                    pos_img_path = random.choice(non_overlap_candidates)
                    break
                    
        if not pos_img_path:
            # 极小概率某个 ID 的图全被 query 抽干了，重新抽题
            return self.get_mixed_gallery_test_case(query_mod, query_size, gallery_size)
            
        gallery_subset.append(pos_img_path)
        
        # 4. 抽取 Gallery 负样本 (混合模态大乱斗) 
        all_neg_images = []
        for pid in self.valid_pids:
            if pid != target_pid:
                for mod in self.modalities:
                    # 获取该 pid 在该模态下的所有图片
                    for p, cam in self.data_dict[mod].get(pid, []):
                        all_neg_images.append(p)
        
        # 计算还需要多少张负样本才能凑够 gallery_size
        num_needed = gallery_size - len(gallery_subset)
        
        if num_needed > 0:
            # 如果池子里的图够多，就进行随机抽样；如果要求的 gallery_size 极端大（超过了底库总数），就全放进去
            actual_needed = min(num_needed, len(all_neg_images))
            if actual_needed < num_needed:
                print(f"⚠️ 警告: 库中负样本总数 ({len(all_neg_images)}) 不足，实际 Gallery 大小将被截断为 {len(gallery_subset) + actual_needed}")
            
            # 从巨大的混合池中随机抽取
            selected_negs = random.sample(all_neg_images, actual_needed)
            gallery_subset.extend(selected_negs)
            
        random.shuffle(gallery_subset)
        gt_indices = []
        for i, path in enumerate(gallery_subset):
            pid, _ = self._parse_filename(path)
            if pid == target_pid:
                gt_indices.append(i)
        
        return query_paths, gallery_subset, gt_indices


if __name__ == "__main__":
    # 请替换为您本地的数据集路径
    DATA_DIR = "/home/user/GSK/heyalan/Reid/data/multi_dataset/RGBNT201/test"
    
    try:
        dataset = RGBNTDataset(DATA_DIR)
        print("\n--- 正在进行多 Query、混合光谱、跨摄像头抽取测试 ---")
        query_mod = "RGB"
        q_paths, g_paths, gt_idx = dataset.get_mixed_gallery_test_case(query_mod=query_mod, query_size=5, gallery_size=100)
        
        print(f"\n[Query 图像 (应全为 {query_mod}，且优先不同 CamID)]:")
        for idx, p in enumerate(q_paths):
            print(f"  {idx+1}: {os.path.basename(p)}")
            
        print(f"\n[Gallery 图像 (大小: {len(g_paths)}, 混合模态)]:")
        for idx, p in enumerate(g_paths):
            mark = "⭐️ [Target!]" if idx in gt_idx else ""
            print(f"  {idx}: {os.path.basename(p)} {mark}")
            
    except Exception as e:
        print(f"测试失败，请检查路径。报错信息: {e}")