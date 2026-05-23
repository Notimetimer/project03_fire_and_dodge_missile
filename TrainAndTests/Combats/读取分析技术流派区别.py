import os
import sys
import numpy as np
from math import *
import pandas as pd
import matplotlib.pyplot as plt

from _context import *
# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# 优先使用dir_name，如果没有则使用experiment_name
dir_name = None # "IL_and_MixedPFSP_分阶段_挑战_并行_分层2s-run-20260408-175230"

# 次要
experiment_name = 'IL_and_Mixed经典PFSP_多技术流派_并行_分层_rule3_0.3'
# --- 查找并加载模型 ---
logs_root_dir = os.path.join(project_root, "logs/combat")


latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
    get_latest_log_dir(logs_root_dir, args.mission_name)

