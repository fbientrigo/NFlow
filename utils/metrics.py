# utils/metrics.py
# Metrics for evaluating model performance
# This file contains functions to compute various metrics such as accuracy, precision, recall, and F1 score.
# KL divergence and Wasserstein distance are also included for distribution comparisons.

# [] Cross-correlations and other statistical measures can be added as needed.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_qq(real, gen, name):
    qs = np.linspace(0,1,100)
    qr = np.quantile(real, qs)
    qg = np.quantile(gen,  qs)
    plt.figure(figsize=(4,4))
    plt.plot(qr, qg, 'o', alpha=0.6)
    m = min(qr.min(), qg.min()); M = max(qr.max(), qg.max())
    plt.plot([m,M],[m,M],'k--')
    plt.xlabel('Real quantiles'); plt.ylabel('Gen quantiles')
    plt.title(f'Q–Q {name}'); plt.tight_layout(); plt.show()



