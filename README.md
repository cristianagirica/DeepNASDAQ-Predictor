# DeepNASDAQ-Predictor

## ✈️ Descrierea Proiectului
Acest proiect este realizat în cadrul Academiei de Studii Economice din București (ASE) și are ca obiectiv utilizarea rețelelor neuronale recurente și hibride pentru prognoza avansată a seriilor de timp financiare. 

Focalizându-se exclusiv pe **sectorul companiilor aeriene listate pe bursa NASDAQ**, proiectul analizează dinamica istorică a prețurilor pe o perioadă de 25 de ani (**2001 - 2026**). Modelele implementate urmăresc să captureze tiparele complexe de volatilitate, sezonalitate și schimbările de regim economic (precum criza din 2008, pandemia din 2020 sau actuala criză a combustibilului) specifice acestei industrii puternic expuse la șocuri externe.

## 📊 Companiile Analizate (Eșantionul Final)
Datele brute sunt procesate cronologic și filtrate longitudinal dintr-un volum masiv de date zilnice NASDAQ. Proiectul include în analiză toate cele 5 companii aeriene identificate structural în setul de date, oferind o acoperire completă a diferitelor modele de business:
* **AAL** (American Airlines Group) – Gigant tradițional (*Legacy Carrier*), cu operațiuni globale masive.
* **UAL** (United Airlines Holdings) – Un alt pilon major al zborurilor internaționale de linie.
* **JBLU** (JetBlue Airways Corp) – Operator reprezentativ pentru segmentul de cost mediu/hibrid.
* **SNCY** (Sun Country Airlines) – Operator cu o dinamică modernă, ultra-low-cost, listat mai recent pe bursă.
* **RJET** (Republic Airways Holdings) – Reprezentant al aviației regionale și de transport secundar.

## ⛽ Obiectiv: Detectarea Impactului Crizei Kerosenului
Pe lângă prognoza clasică a prețurilor, un pilon central și inovator al acestui proiect este evaluarea capacității rețelelor neuronale de a anticipa și detecta impactul **crizei actuale a kerosenului** asupra acțiunilor de profil. Combustibilul reprezintă cea mai mare cheltuială operațională din aviație (25% - 40%). Proiectul testează dacă modelele de Deep Learning pot prezice:
1. **Schimbările bruște de regim:** Anticiparea exploziilor de volatilitate și a corecțiilor de preț declanșate de șocurile de pe piața energetică.
2. **Decuplarea performanței:** Modul în care criza afectează diferențiat giganții internaționali (`UAL`, `AAL`) față de operatorii regionali sau de nișă (`JBLU`, `SNCY`, `RJET`).

---

## 📂 Structura Directorului Proiectului

Proiectul este organizat modular pentru a separa în mod clar datele de intrare, componentele infrastructurii de antrenare, modelele finale serializate și livrabilele analitice/grafice.

```text
DeepNASDAQ-Predictor/
│
├── dataIn/                             # Datele de intrare (Brute și Intermediare)
│   ├── NASDAQ_companii_aeriene_2001-2026.csv   # Setul de date istoric original (5 companii)
│   ├── features.csv                    # Generat la Pasul 1 (conține indicatorii tehnici calculați)
│   └── train/                          # [Creat la Pasul 2] Matricile binare NumPy (.npy)
│       ├── X_train.npy                 # 80% din date per companie (ferestre temporale)
│       ├── y_train.npy                 # Variabila țintă de antrenare (TARGET)
│       ├── X_test.npy                  # 20% din datele recente per companie
│       └── y_test.npy                  # Variabila țintă de testare
│
├── src/                                # Fișierele codului sursă Python (.py)
│   ├── 01_ingineria_atributelor.py  # Pasul 1: Calcul indicatori (SMA, MACD, Bollinger, Volatilitate)
│   ├── 02_pregatire_date.py             # Pasul 2: Split cronologic și transformare MinMaxScaler
│   ├── 03_model_lstm.py               # Pasul 3: Antrenare arhitectură LSTM Standard Custom
│   ├── 04_model_bilstm.py             # Pasul 4: Antrenare arhitectură Bidirecțională (BiLSTM)
│   ├── 05_model_cnn_lstm.py           # Pasul 5: Antrenare arhitectură hibridă Conv1D + LSTM
│   ├── 06_evaluare.py         # Pasul 6: Evaluare pe setul de test, metrici USD și ploturi reale
│   ├── 07_prognoza.py            # Pasul 7: Bucla autoregresivă pe 252 pași cu recalculare indicatori
│   └── nn.py                              # Wrapper Generic: Pipeline personalizat ModelTrainer cu GradientTape
│
├── util/                               # Obiecte utilitare, scalere și puncte de control intermediare
│   ├── scaler_x.pkl                    # Scalerul caracteristicilor geometrice salvat de Pasul 2
│   ├── scaler_y.pkl                    # Scalerul etichetei prețului salvat de Pasul 2
│   ├── lstm_model_best.keras           # Checkpoint pentru cel mai bun val_loss (LSTM)
│   ├── bilstm_model_best.keras         # Checkpoint pentru cel mai bun val_loss (BiLSTM)
│   └── cnn_lstm_model_best.keras       # Checkpoint pentru cel mai bun val_loss (CNN-LSTM)
│
├── model/                              # Modelele finale optimizate și salvate pe disc (.keras)
│   ├── lstm_model.keras
│   ├── bilstm_model.keras
│   └── cnn_lstm_model.keras
│
├── dataOut/                            # Fișiere de ieșire rezultate din execuție
│   ├── lstm_predictions_real.csv       # Predicții brute brute la scară reală (USD)
│   ├── bilstm_predictions_real.csv
│   ├── cnn_lstm_predictions_real.csv
│   │
│   ├── evaluare/                       # Rezultatele analizei comparative pe setul de test (Pasul 6)
│   │   ├── LSTM_predictions_real.csv
│   │   ├── BiLSTM_predictions_real.csv
│   │   ├── CNN_LSTM_predictions_real.csv
│   │   └── plot/                       # Cele 15 grafice comparative de test per model/companie (USD)
│   │
│   ├── prognoza/                       # Rezultatele simulării pe un an în viitor (Pasul 7)
│   │   ├── prognoza_dinamica_LSTM.csv  # Traiectorii prognozate dinamic pe 252 de zile
│   │   ├── prognoza_dinamica_BiLSTM.csv
│   │   ├── prognoza_dinamica_CNN_LSTM.csv
│   │   └── plot/                       # Cele 15 grafice de evoluție nelineară viitoare
│   │
│   └── plot/                           # Grafice de diagnostic și performanță brute
│       ├── lstm_loss_curve.png         # Curbele de eroare (Train vs Val Loss) per model
│       ├── bilstm_loss_curve.png
│       └── cnn_lstm_loss_curve.png
│ 
└── preprocesare/                       
│   └── preprocesare.py  # Structurarea si filtrarea seturilor de date initiale in setul de date final dorit
│
└── docs/                               # Documentația oficială a proiectului academic
    └── Proiect_Retele_Neuronale_Echipa.docx  # Justificare economică, metodologie și interpretarea metricilor