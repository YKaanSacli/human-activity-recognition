import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import shap
    import matplotlib.pyplot as plt

    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        d1 = pd.read_csv("train.csv")
        d2 = pd.read_csv("test.csv")
        
        le = LabelEncoder()
        d1["Activity"] = le.fit_transform(d1["Activity"])
        
        x_tr = d1.drop(["Activity", "subject"], axis=1, errors="ignore")
        y_tr = d1["Activity"]
        
        x_te = d2.drop(["Activity", "subject"], axis=1, errors="ignore")
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(x_tr, y_tr)
        
        explainer = shap.TreeExplainer(rf)
        sample = x_te.iloc[:30]
        shap_vals = explainer.shap_values(sample)
        
        shap.summary_plot(shap_vals, sample, show=False)
        plt.tight_layout()
        plt.savefig("shap_ozet.png")
        print("shap_ozet.png kaydedildi")
    else:
        print("train.csv test.csv yok")
except ImportError:
    print("shap kurulu degil")
