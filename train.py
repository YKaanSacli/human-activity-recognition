import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse

p = argparse.ArgumentParser()
p.add_argument("--model", type=str, default="svm")
p.add_argument("--save", action="store_true")
args = p.parse_args()

if os.path.exists("train.csv") and os.path.exists("test.csv"):
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    
    lbl = LabelEncoder()
    df_train["Activity"] = lbl.fit_transform(df_train["Activity"])
    df_test["Activity"] = lbl.transform(df_test["Activity"])
    
    xtr = df_train.drop(["Activity", "subject"], axis=1, errors="ignore")
    ytr = df_train["Activity"]
    
    xte = df_test.drop(["Activity", "subject"], axis=1, errors="ignore")
    yte = df_test["Activity"]
    
    sc = StandardScaler()
    xtr_sc = sc.fit_transform(xtr)
    xte_sc = sc.transform(xte)
    
    secilen = args.model.lower()
    
    if secilen == "svm":
        m = SVC(probability=True)
    elif secilen == "rf":
        m = RandomForestClassifier(n_estimators=100)
    elif secilen == "lr":
        m = LogisticRegression(max_iter=1000)
    else:
        m = SVC(probability=True)
        
    print("model egitiliyor:", secilen)
    m.fit(xtr_sc, ytr)
    
    pred = m.predict(xte_sc)
    acc = accuracy_score(yte, pred)
    print("dogruluk:", acc)
    print(classification_report(yte, pred))
    
    if args.save:
        if not os.path.exists("kaydedilen_modeller"):
            os.makedirs("kaydedilen_modeller")
        joblib.dump(m, f"kaydedilen_modeller/{secilen}_model.pkl")
        joblib.dump(sc, "kaydedilen_modeller/scaler.pkl")
        joblib.dump(lbl, "kaydedilen_modeller/label_encoder.pkl")
        print("kaydedildi")
else:
    print("train.csv veya test.csv bulunamadi calistirilamiyor")
