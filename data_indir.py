import os
import urllib.request
import zipfile

url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
dosya = "uci_har.zip"

if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
    print("indiriliyor...")
    try:
        urllib.request.urlretrieve(url, dosya)
        with zipfile.ZipFile(dosya, 'r') as z:
            z.extractall("uci_temp")
        inner_zip = "uci_temp/UCI HAR Dataset.zip"
        if os.path.exists(inner_zip):
            with zipfile.ZipFile(inner_zip, 'r') as z:
                z.extractall("uci_temp")
        print("zip acildi")
    except Exception as e:
        print("hata:", e)
else:
    print("csv dosyalari zaten var.")
