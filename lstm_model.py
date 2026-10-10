import pandas as pd
import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        d1 = pd.read_csv("train.csv")
        d2 = pd.read_csv("test.csv")
        
        le = LabelEncoder()
        y1 = le.fit_transform(d1["Activity"])
        y2 = le.transform(d2["Activity"])
        
        x1 = d1.drop(["Activity", "subject"], axis=1, errors="ignore").values
        x2 = d2.drop(["Activity", "subject"], axis=1, errors="ignore").values
        
        sc = StandardScaler()
        x1 = sc.fit_transform(x1)
        x2 = sc.transform(x2)
        
        tx1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(1)
        ty1 = torch.tensor(y1, dtype=torch.long)
        tx2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(1)
        ty2 = torch.tensor(y2, dtype=torch.long)
        
        ldr = DataLoader(TensorDataset(tx1, ty1), batch_size=64, shuffle=True)
        
        class BasitLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=561, hidden_size=64, batch_first=True)
                self.fc = nn.Linear(64, len(np.unique(y1)))
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
                
        net = BasitLSTM()
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        
        print("lstm basliyor...")
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
            print("lstm test dogrulugu:", acc.item())
    else:
        print("train.csv test.csv yok")
except ImportError:
    print("torch yok")
