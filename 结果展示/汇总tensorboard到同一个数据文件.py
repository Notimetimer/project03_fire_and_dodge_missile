import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from _context import * # 包含 project_root

# --- 1. 环境与绘图配置 ---
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

OriginalData_dir = os.path.join(project_root, "logs", "OriginalData")
Data_dir = os.path.join(project_root, "logs", "Data")

