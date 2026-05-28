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
| Total Queries | 68 |
| Avg Rank-1 Accuracy | **21.00%** |
| Avg Rank-3 Accuracy | **36.00%** |
| Overall Rank-1 Hits | 15 / 68 |
| Overall Rank-3 Hits | 26 / 68 |
| Elapsed Time | 719.0s |

## Per-Trial Results

| Trial | Person ID | Queries | Rank-1 | Rank-3 | R1 Acc | R3 Acc |
|---|---|---|---|---|---|---|
| 1 | 631 | 5 | 2 | 4 | 40% | 80% |
| 2 | 691 | 5 | 1 | 1 | 20% | 20% |
| 3 | 728 | 5 | 0 | 2 | 0% | 40% |
| 4 | 567 | 5 | 1 | 1 | 20% | 20% |
| 5 | 1249 | 5 | 1 | 2 | 20% | 40% |
| 6 | 1015 | 4 | 1 | 1 | 25% | 25% |
| 7 | 989 | 4 | 1 | 1 | 25% | 25% |
| 8 | 687 | 5 | 0 | 1 | 0% | 20% |
| 9 | 1046 | 3 | 0 | 0 | 0% | 0% |
| 10 | 1490 | 5 | 2 | 2 | 40% | 40% |
| 11 | 770 | 5 | 0 | 1 | 0% | 20% |
| 12 | 1255 | 5 | 1 | 3 | 20% | 60% |
| 13 | 1036 | 3 | 0 | 0 | 0% | 0% |
| 14 | 447 | 4 | 1 | 2 | 25% | 50% |
| 15 | 625 | 5 | 4 | 5 | 80% | 100% |
