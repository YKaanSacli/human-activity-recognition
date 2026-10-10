import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

if os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
    lbl = LabelEncoder()
    y = lbl.fit_transform(df["Activity"])
    x = df.drop(["Activity", "subject"], axis=1, errors="ignore")
    
    param = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    arama = RandomizedSearchCV(rf, param, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    
    print("parametre aramasi yapiliyor...")
    arama.fit(x, y)
    
    print("en iyi parametreler:", arama.best_params_)
    print("en iyi skor:", arama.best_score_)
else:
    print("train.csv yok")
