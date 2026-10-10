import pandas as pd
import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        t1 = pd.read_csv("train.csv")
        t2 = pd.read_csv("test.csv")
        
        le = LabelEncoder()
        y1 = le.fit_transform(t1["Activity"])
        y2 = le.transform(t2["Activity"])
        
        x1 = t1.drop(["Activity", "subject"], axis=1, errors="ignore").values
        x2 = t2.drop(["Activity", "subject"], axis=1, errors="ignore").values
        
        s = StandardScaler()
        x1 = s.fit_transform(x1)
        x2 = s.transform(x2)
        
        tx1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(1)
        ty1 = torch.tensor(y1, dtype=torch.long)
        tx2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(1)
        ty2 = torch.tensor(y2, dtype=torch.long)
        
        ldr = DataLoader(TensorDataset(tx1, ty1), batch_size=64, shuffle=True)
        
        class BasitCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.r1 = nn.ReLU()
                self.p1 = nn.MaxPool1d(2)
                self.c2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.r2 = nn.ReLU()
                self.p2 = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(64, len(np.unique(y1)))
                
            def forward(self, x):
                x = self.p1(self.r1(self.c1(x)))
                x = self.p2(self.r2(self.c2(x)))
                x = x.squeeze(2)
                return self.fc(x)
                
        net = BasitCNN()
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        
        print("cnn egitimi basliyor...")
        for ep in range(5):
            for bx, by in ldr:
                opt.zero_grad()
                out = net(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
            print("epoch:", ep+1, "loss:", loss.item())
            
        with torch.no_grad():
            preds = net(tx2).argmax(dim=1)
            acc = (preds == ty2).float().mean()
            print("cnn test dogrulugu:", acc.item())
    else:
        print("csv dosyalari yok")
except ImportError:
    print("torch yuklu degil, calismadi")
