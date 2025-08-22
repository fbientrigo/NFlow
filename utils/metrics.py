# utils/metrics.py
# Metrics for evaluating model performance
# This file contains functions to compute various metrics such as accuracy, precision, recall, and F1 score.
# KL divergence and Wasserstein distance are also included for distribution comparisons.

# [] Cross-correlations and other statistical measures can be added as needed.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

