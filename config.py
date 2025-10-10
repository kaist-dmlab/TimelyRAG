import os
from datetime import datetime

# Experiment directory
EXP_DIR = os.environ.get(
    "TR_EXPERIMENT_DIR",
    "./test/results",
)
os.makedirs(EXP_DIR, exist_ok=True)

# Evaluation / retrieval settings
K = 10
STAGE1_TOPN = 10 * K

# Alpha grid & normalization hyperparameters
ALPHA_GRID   = [0.0] + [i / 10.0 for i in range(1, 10)] + [1.0]
TIME_STD_MAX = 0.35
TEXT_STD_MAX = 0.35
MISS_MARGIN  = 0.05

# GapAvg upper caps by granularity (tunable)
GAP_MAX = {
    "hour":  72.0,   # 3 days
    "day":   60.0,   # 2 months
    "month": 48.0,   # 4 years
    "year":  50.0,   # 50 years
}

# Dataset paths
DATASET_PATHS = [
    "./datasets/timelyrag_company_policies_dataset_v2.jsonl",
    "./datasets/timelyrag_law_dataset_v2.jsonl",
    "./datasets/timelyrag_terms_of_service_dataset_v2.jsonl",
    "./datasets/timelyrag_yonsei_dataset_v2.jsonl",
]

# Logging helper
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = f"{EXP_DIR}/retrieval_results_{TS}.log"