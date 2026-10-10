import pandas as pd
import numpy as np
import joblib
import sys
import os

if len(sys.argv) > 1:
    dosya_yolu = sys.argv[1]
else:
    dosya_yolu = "test.csv"

if os.path.exists("kaydedilen_modeller/svm_model.pkl") and os.path.exists(dosya_yolu):
    m = joblib.load("kaydedilen_modeller/svm_model.pkl")
    sc = joblib.load("kaydedilen_modeller/scaler.pkl")
    lbl = joblib.load("kaydedilen_modeller/label_encoder.pkl")
    
    veri = pd.read_csv(dosya_yolu)
    x = veri.drop(["Activity", "subject"], axis=1, errors="ignore")
    x_olcekli = sc.transform(x.iloc[[0]])
    sonuc = m.predict(x_olcekli)
    print("tahmin:", lbl.inverse_transform(sonuc)[0])
else:
    print("model veya veri dosyasi bulunamadi")
