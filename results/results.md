# Generated results

### Test window 2003-2005 (top 1% cut-off)

| Model | AUC | NDCG@k | Precision top 1% | Frauds caught (sum over years) | Paper AUC | Paper NDCG@k | Paper precision | Paper caught |
|---|---|---|---|---|---|---|---|---|
| Beneish M-score (fixed 1999 weights) | 0.554 | 0.011 | 1.12% | 2 of 172 | - | - | - | - |
| Logit, 14 ratios (Dechow-style) | 0.671 | 0.000 | 0.00% | 0 of 172 | 0.648 | 0.011 | 1.28% | 2 |
| Logit, 28 raw items | 0.705 | 0.037 | 4.49% | 8 of 172 | 0.674 | 0.008 | 1.13% | 2 |
| RUSBoost, 28 raw items | 0.733 (0.726-0.738) | 0.032 | 3.69% | 6.6 (6-8) of 172 | 0.743 | 0.039 | 5.02% | 9 |
| LightGBM, 28 raw items | 0.718 | 0.034 | 3.93% | 7 of 172 | - | - | - | - |
| Small neural net, 28 raw items | 0.704 | 0.025 | 2.80% | 5 of 172 | - | - | - | - |

Top 1% = 178 firm-years flagged in total across the window.

### Test window 2003-2008 (top 1% cut-off)

| Model | AUC | NDCG@k | Precision top 1% | Frauds caught (sum over years) | Paper AUC | Paper NDCG@k | Paper precision | Paper caught |
|---|---|---|---|---|---|---|---|---|
| Beneish M-score (fixed 1999 weights) | 0.563 | 0.016 | 1.14% | 4 of 261 | - | - | - | - |
| Logit, 14 ratios (Dechow-style) | 0.691 | 0.030 | 1.98% | 7 of 261 | 0.671 | 0.027 | 2.62% | 8 |
| Logit, 28 raw items | 0.673 | 0.022 | 2.54% | 9 of 261 | 0.684 | 0.004 | 0.57% | 2 |
| RUSBoost, 28 raw items | 0.715 (0.710-0.718) | 0.017 | 1.96% | 7.0 (6-9) of 261 | 0.723 | 0.024 | 2.81% | 10 |
| LightGBM, 28 raw items | 0.696 | 0.027 | 2.26% | 8 of 261 | - | - | - | - |
| Small neural net, 28 raw items | 0.682 | 0.016 | 1.70% | 6 of 261 | - | - | - | - |

Top 1% = 352 firm-years flagged in total across the window.

### Test window 2003-2014 (top 1% cut-off)

| Model | AUC | NDCG@k | Precision top 1% | Frauds caught (sum over years) | Paper AUC | Paper NDCG@k | Paper precision | Paper caught |
|---|---|---|---|---|---|---|---|---|
| Beneish M-score (fixed 1999 weights) | 0.533 | 0.008 | 0.57% | 4 of 373 | - | - | - | - |
| Logit, 14 ratios (Dechow-style) | 0.721 | 0.023 | 1.59% | 11 of 373 | 0.702 | 0.022 | 1.85% | 11 |
| Logit, 28 raw items | 0.705 | 0.018 | 1.73% | 12 of 373 | 0.706 | 0.010 | 0.90% | 6 |
| RUSBoost, 28 raw items | 0.724 (0.720-0.729) | 0.018 | 1.53% | 10.6 (7-14) of 373 | 0.720 | 0.018 | 1.87% | 13 |
| LightGBM, 28 raw items | 0.698 | 0.023 | 1.75% | 12 of 373 | - | - | - | - |
| Small neural net, 28 raw items | 0.689 | 0.018 | 1.47% | 10 of 373 | - | - | - | - |

Top 1% = 682 firm-years flagged in total across the window.

### Serial-fraud ablation (top 1%)

| Model | Window | AUC with fix | AUC without fix | Caught with fix | Caught without fix |
|---|---|---|---|---|---|
| RUSBoost, 28 raw items | 2003-2005 | 0.733 | 0.803 | 6.6 | 17.8 |
| RUSBoost, 28 raw items | 2003-2008 | 0.715 | 0.778 | 7.0 | 18.4 |
| RUSBoost, 28 raw items | 2003-2014 | 0.724 | 0.779 | 10.6 | 25.4 |
| LightGBM, 28 raw items | 2003-2005 | 0.718 | 0.786 | 7.0 | 13.0 |
| LightGBM, 28 raw items | 2003-2008 | 0.696 | 0.735 | 8.0 | 14.0 |
| LightGBM, 28 raw items | 2003-2014 | 0.698 | 0.731 | 12.0 | 17.0 |

### Beneish threshold (M > -1.78), 2003-2008

Flags 5,290 of 35,166 firm-years (15.0%); precision 0.81% vs base rate 0.74%; recall 16.5%.

### Explained examples (LightGBM TreeSHAP)

**top_scored**: gvkey 16477, fiscal 2006, AAER label 0, score 0.898, rank 1 of 5908, M-score -2.46

| Item | Value ($m) | Year median | SHAP (log-odds) |
|---|---|---|---|
| act | 3,890.300 | 67.038 | +2.269 |
| sstk | 199.400 | 2.213 | +0.741 |
| prcc_f | 29.530 | 8.480 | +0.284 |
| ap | 2,317.400 | 8.568 | +0.271 |
| ppegt | 2,785.900 | 48.092 | -0.248 |
| invt | 581.500 | 3.516 | +0.228 |
| ib | -710.400 | 1.686 | +0.196 |
| dp | 392.200 | 4.682 | -0.191 |

**top_true_fraud**: gvkey 238616, fiscal 2008, AAER label 1, score 0.820, rank 3 of 5612, M-score -2.59

| Item | Value ($m) | Year median | SHAP (log-odds) |
|---|---|---|---|
| act | 26,563.438 | 73.987 | +1.169 |
| lct | 25,436.137 | 34.629 | +0.974 |
| sstk | 75.441 | 0.513 | +0.767 |
| ppegt | 27,130.219 | 57.709 | -0.726 |
| cogs | 44,841.172 | 73.987 | +0.505 |
| csho | 457.597 | 40.989 | +0.500 |
| rect | 18,579.916 | 16.229 | +0.184 |
| re | -5,709.156 | -6.037 | +0.164 |


Run: 14.7 min, peak RSS 582 MB, Python 3.14.4.
