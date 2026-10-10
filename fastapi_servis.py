import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class GirdiVerisi(BaseModel):
    sensor_degerleri: List[float]

@app.get("/")
def ana():
    return {"durum": "aktif"}

@app.post("/tahmin")
def tahmin_et(veri: GirdiVerisi):
    if os.path.exists("kaydedilen_modeller/svm_model.pkl") and os.path.exists("kaydedilen_modeller/scaler.pkl"):
        m = joblib.load("kaydedilen_modeller/svm_model.pkl")
        sc = joblib.load("kaydedilen_modeller/scaler.pkl")
        lbl = joblib.load("kaydedilen_modeller/label_encoder.pkl")
        
        arr = [veri.sensor_degerleri]
        arr_sc = sc.transform(arr)
        pred = m.predict(arr_sc)
        isim = lbl.inverse_transform(pred)[0]
        return {"tahmin": isim}
    return {"hata": "model dosyalari yok"}
