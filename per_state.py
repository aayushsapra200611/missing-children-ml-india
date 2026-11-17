#!/usr/bin/env python3
"""
per_state.py — Per-State/UT Test Error Audit (Appendix A, Table A1)
Subject: Machine Learning | Course: B.Sc. (Hons.) Computer Science, 3rd Year
Author: Aayush Sapra | Roll No.: 24570096 | Supervisor: Nikhil Sir
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_experiments import (
    train_df, test_df, NUMERIC_COLS, Preprocessor, Poly2RidgeModel, RidgeModel
)

prep = Preprocessor(use_pca=False).fit(train_df)
X_tr = prep.transform(train_df)
X_te = prep.transform(test_df)
y_tr = train_df["target_missing_next"].values

m_poly = Poly2RidgeModel(alpha=10.0, n_num=len(NUMERIC_COLS)).fit(X_tr, y_tr)
m_ridge = RidgeModel(alpha=0.1).fit(X_tr, y_tr)

test_eval = test_df.copy()
test_eval["pred_poly"] = np.maximum(0, m_poly.predict(X_te))
test_eval["pred_ridge"] = np.maximum(0, m_ridge.predict(X_te))
test_eval["pred_pers"] = test_eval["curr_missing_t"]

rows = []
for st in sorted(test_eval["state"].unique()):
    n_tr = int((train_df["state"] == st).sum())
    sub = test_eval[test_eval["state"] == st]
    n_te = len(sub)
    mean_miss = float(sub["target_missing_next"].mean())
    rmse_poly = float(np.sqrt(np.mean((sub["target_missing_next"] - sub["pred_poly"]) ** 2)))
    rmse_pers = float(np.sqrt(np.mean((sub["target_missing_next"] - sub["pred_pers"]) ** 2)))
    rows.append({
        "State/UT": st,
        "n_train": n_tr,
        "n_test": n_te,
        "Mean Missing": round(mean_miss, 1),
        "RMSE Poly-2": round(rmse_poly, 1),
        "RMSE Pers.": round(rmse_pers, 1),
        "Worse": rmse_poly > rmse_pers
    })

df_out = pd.DataFrame(rows).sort_values("Mean Missing", ascending=False)
print(df_out.to_string(index=False))