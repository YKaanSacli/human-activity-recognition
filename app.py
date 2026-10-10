import os
import pandas as pd
import numpy as np
import joblib

try:
    import streamlit as st

    st.title("Human Activity Recognition Test Arayuzu")

    if os.path.exists("kaydedilen_modeller/svm_model.pkl") and os.path.exists("kaydedilen_modeller/scaler.pkl"):
        model = joblib.load("kaydedilen_modeller/svm_model.pkl")
        scaler = joblib.load("kaydedilen_modeller/scaler.pkl")
        lbl = joblib.load("kaydedilen_modeller/label_encoder.pkl")
        
        st.write("Model yuklendi.")
        
        if os.path.exists("test.csv"):
            test_df = pd.read_csv("test.csv")
            satir_no = st.number_input("Test verisinden satir sec:", min_value=0, max_value=len(test_df)-1, value=0)
            
            secilen_satir = test_df.iloc[[satir_no]]
            st.write("Gercek Aktivite:", secilen_satir["Activity"].values[0] if "Activity" in secilen_satir else "Bilinmiyor")
            
            x_giris = secilen_satir.drop(["Activity", "subject"], axis=1, errors="ignore")
            x_olcekli = scaler.transform(x_giris)
            
            if st.button("Tahmin Et"):
                tahmin = model.predict(x_olcekli)
                tahmin_isim = lbl.inverse_transform(tahmin)[0]
                st.success(f"Tahmin Edilen Aktivite: {tahmin_isim}")
                
                if hasattr(model, "predict_proba"):
                    olasiliklar = model.predict_proba(x_olcekli)[0]
                    siniflar = lbl.classes_
                    prob_df = pd.DataFrame({"Aktivite": siniflar, "Olasilik": olasiliklar})
                    st.bar_chart(prob_df.set_index("Aktivite"))
        else:
            st.warning("test.csv bulunamadi.")
    else:
        st.info("Oncelikle train.py --model svm --save calistirilmali model dosyalari yok.")
except ImportError:
    print("streamlit yok")
