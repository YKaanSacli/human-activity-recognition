import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
    y = df["Activity"]
    x = df.drop(["Activity", "subject"], axis=1, errors="ignore")
    
    sc = StandardScaler()
    x_sc = sc.fit_transform(x)
    
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_sc)
    
    plt.figure(figsize=(9, 6))
    for sinif in y.unique():
        ind = (y == sinif)
        plt.scatter(x_pca[ind, 0], x_pca[ind, 1], label=sinif, alpha=0.5, s=15)
        
    plt.title("PCA 2 Boyutlu Gorsellestirme")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_grafigi.png")
    print("pca_grafigi.png olusturuldu")
else:
    print("train.csv yok")
