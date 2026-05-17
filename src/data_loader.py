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

    def get_test_case(self, gallery_size=10, mode="random"):
        """
        Returns a single test case:
        - query_path: str
        - gallery_paths: List[str] (contains at least one match)
        - ground_truth_idx: int (index of the match in gallery_paths)
        """
        # 1. Pick a random query
        query_path = random.choice(self.query_paths)
        q_pid, q_camid = self._parse_filename(query_path)
        
        # 2. Find positives in gallery
        gallery_full = [(p, *self._parse_filename(p)) for p in self.gallery_paths]
        
        # Standard ReID protocol: match is same ID, different camera usually.
        # But Market-1501 might allow same camera if time is different? 
        # For simplicity in this LMM demo strict protocol:
        # positive: same pid, different camid? Or just same pid?
        # Market-1501 evaluation usually ignores "junk" (same ID, same camera). 
        # But here we just want to find "same person". LMMs might be good at strict matches too.
        # Let's simple filter: same pid is positive.
        
        positives = [p for (p, pid, cam) in gallery_full if pid == q_pid]
        negatives = [p for (p, pid, cam) in gallery_full if pid != q_pid]
        
        if not positives:
            # Retry if no match found for this query (should differ rarely)
            return self.get_test_case(gallery_size, mode)
            
        # 3. Sample
        # Ensure we have at least one positive
        target_positive = random.choice(positives)
        
        # Fill rest with negatives
        num_negatives = gallery_size - 1
        curr_negatives = random.sample(negatives, min(len(negatives), num_negatives))
        
        # Combine
        gallery_subset = [target_positive] + curr_negatives
        random.shuffle(gallery_subset)
        
        # Find index
        gt_idx = gallery_subset.index(target_positive)
        
        # Check if there are accidental other positives in the negatives selection (unlikely if sampled from strict negatives)
        # But if we just sampled from 'all', we'd need to check. here we sampled from 'negatives', so pid is diff.
        
        return query_path, gallery_subset, gt_idx
