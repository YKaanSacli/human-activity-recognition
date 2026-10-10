import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_scaler_islemi():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    s = StandardScaler()
    res = s.fit_transform(x)
    assert res.shape == (3, 2)
    assert round(res.mean()) == 0

def test_dosya_var_mi():
    assert os.path.exists("requirements.txt")
    assert os.path.exists("human-activity-recognition.ipynb")
