# Batch ReID Evaluation Report

## Config

- **data_dir**: /mnt/l/CV/data/1/Market-1501-v15.09.15
- **gallery_size**: 50
- **query_size**: 5
- **num_trials**: 15
- **retrieval_mode**: LangChain CLIP Vector DB

## Summary

| Metric | Value |
|---|---|
| Total Trials | 15 |
| Total Queries | 67 |
| Avg Rank-1 Accuracy | **31.78%** |
| Avg Rank-3 Accuracy | **37.44%** |
| Overall Rank-1 Hits | 21 / 67 |
| Overall Rank-3 Hits | 25 / 67 |
| Elapsed Time | 727.1s |

## Per-Trial Results

| Trial | Person ID | Queries | Rank-1 | Rank-3 | R1 Acc | R3 Acc |
|---|---|---|---|---|---|---|
| 1 | 853 | 4 | 1 | 1 | 25% | 25% |
| 2 | 1082 | 5 | 1 | 2 | 20% | 40% |
| 3 | 356 | 3 | 1 | 1 | 33% | 33% |
| 4 | 1423 | 3 | 1 | 1 | 33% | 33% |
| 5 | 1015 | 4 | 3 | 4 | 75% | 100% |
| 6 | 3 | 5 | 0 | 1 | 0% | 20% |
| 7 | 146 | 5 | 0 | 0 | 0% | 0% |
| 8 | 161 | 5 | 2 | 2 | 40% | 40% |
| 9 | 1175 | 5 | 0 | 0 | 0% | 0% |
| 10 | 320 | 5 | 3 | 3 | 60% | 60% |
| 11 | 305 | 5 | 1 | 1 | 20% | 20% |
| 12 | 154 | 5 | 4 | 4 | 80% | 80% |
| 13 | 1050 | 4 | 1 | 1 | 25% | 25% |
| 14 | 1328 | 4 | 1 | 1 | 25% | 25% |
| 15 | 797 | 5 | 2 | 3 | 40% | 60% |
