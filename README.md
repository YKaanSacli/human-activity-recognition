# 📱 Human Activity Recognition (HAR) Using Smartphone Sensor Data
### Akıllı Telefon Sensör Verileri ile İnsan Aktivitelerinin Sınıflandırılması

---

## 📌 Project Overview / Proje Özeti

* **EN:** This project classifies six daily human activities using tri-axial accelerometer and gyroscope data from smartphones by benchmarking nine supervised machine learning algorithms].
* **TR:** Bu proje, akıllı telefonlardan toplanan üç eksenli ivmeölçer ve jiroskop verilerini kullanarak altı günlük insan aktivitesini dokuz farklı denetimli makine öğrenmesi algoritması ile sınıflandırıp karşılaştırır.

---

## 📊 Dataset Specifications / Veri Seti Özellikleri

### EN - English
* **Benchmark Dataset:** UCI Human Activity Recognition (HAR)
* **Subjects:** 30 volunteers
* **Sampling Rate:** 50 Hz
* **Extracted Features:** 561 time and frequency domain variables
* **Target Classes:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying

### TR - Türkçe
* **Kullanılan Veri Seti:** UCI Human Activity Recognition (HAR)
* **Katılımcı Sayısı:** 30 gönüllü
* **Örnekleme Frekansı:** 50 Hz
* **Çıkarılan Öznitelik Sayısı:** 561 adet zaman ve frekans alanı değişkeni
* **Hedef Aktiviteler:** Yürüme, Merdiven Çıkma, Merdiven İnme, Oturma, Ayakta Durma, Uzanma

---

## 🤖 Model Comparison / Model Karşılaştırma Sonuçları

| Model | Accuracy / Doğruluk | Training Time / Eğitim Süresi |
|---|---:|---:|
| Logistic Regression / Lojistik Regresyon | 95.52% | 2.65 s |
| Support Vector Machines (SVM) | 95.18% | 2.02 s |
| Gradient Boosting | 93.99% | 583.94 s |
| Extra Trees | 93.96% | 2.05 s] |
| Random Forest / Rastgele Orman | 92.60% | 9.36 s |
| QDA | 92.43% | 0.80 s |
| KNN | 88.36% | 0.02 s |
| Decision Tree / Karar Ağacı | 85.44% | 3.59 s |
| AdaBoost | 34.92% | 20.21 s |

---

## 🛠️ Technologies & Tools / Teknolojiler ve Araçlar

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---
## 🚀 Installation & Execution / Kurulum ve Çalıştırma

```bash
# 1. Clone repository / Depoyu klonlayın
git clone [https://github.com/YKaanSacli/human-activity-recognition.git](https://github.com/YKaanSacli/human-activity-recognition.git)

# 2. Enter directory / Klasöre girin
cd human-activity-recognition

# 3. Install dependencies / Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Run notebook / Notebook'u başlatın
jupyter notebook
```
## 🚀 Installation & Execution / Kurulum ve Çalıştırma

```bash
# 1. Clone repository / Depoyu klonlayın
git clone https://github.com/YKaanSacli/human-activity-recognition.git

# 2. Enter directory / Klasöre girin
cd human-activity-recognition

# 3. Install dependencies / Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Run notebook / Notebook'u başlatın
jupyter notebook
```

---

## 📁 Project Structure / Proje Yapısı

```text
human-activity-recognition/
│
├── human-activity-recognition.ipynb   # Main notebook / Ana notebook
├── requirements.txt                   # Dependencies / Kütüphane listesi
├── LICENSE                            # MIT License
└── README.md                          # Documentation / Proje tanıtımı
```

---

## 👨‍‍💻 Author / Hazırlayan
* **Yusuf Kaan Saçlı**

### 🎓 Education / Eğitim
* **Master's Degree (Ongoing) / Yüksek Lisans (Devam Ediyor):** Artificial Intelligence and Data Science, Sivas Cumhuriyet University / Sivas Cumhuriyet Üniversitesi, Yapay Zeka ve Veri Bilimi Tezli Yüksek Lisans
* **Bachelor's Degree / Lisans:** Statistics & Computer Science (2026), Sivas Cumhuriyet University / Sivas Cumhuriyet Üniversitesi, İstatistik ve Bilgisayar Bilimleri (2026)

### 🎯 Thesis Topic / Tez Konusu
* Human Activity Recognition (HAR) Using Smartphone Sensor Data & Machine Learning / Akıllı Telefon Sensör Verileri Kullanılarak Makine Öğrenmesi ile İnsan Aktivitelerinin Sınıflandırılması

### 🔗 Profiles / Profiller
* **GitHub:** [YKaanSacli](https://github.com/YKaanSacli)
* **LinkedIn:** [yusufkaansacli](https://www.linkedin.com/in/yusufkaansacli)

---

## 📄 License / Lisans
This project is licensed under the MIT License / Bu proje MIT Lisansı ile lisanslanmıştır.
