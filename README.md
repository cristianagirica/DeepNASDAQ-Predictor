# DeepNASDAQ-Predictor

## Descrierea Proiectului
Acest proiect este realizat în cadrul Academiei de Studii Economice din București (ASE) și are ca obiectiv utilizarea rețelelor neuronale recurente pentru prognoza seriilor de timp financiare. 

Focalizându-se exclusiv pe **sectorul companiilor aeriene listate pe bursa NASDAQ**, proiectul analizează dinamica istorică a prețurilor pe o perioadă de 25 de ani (**2001 - 2026**). Modelele implementate urmăresc să captureze tiparele complexe de volatilitate, sezonalitate și schimbările de regim economic (precum criza din 2008 sau pandemia din 2020) specifice acestei industrii puternic expuse la șocuri externe.

## Companiile Analizate (Eșantionul Final)
Datele brute sunt procesate cronologic și filtrate longitudinal dintr-un volum masiv de fișiere CSV zilnice. Proiectul include în analiză toate cele 5 companii aeriene identificate structural în setul de date NASDAQ, oferind o acoperire completă a diferitelor modele de business:
* **AAL** (American Airlines Group) – Gigant tradițional (*Legacy Carrier*), cu operațiuni globale masive.
* **UAL** (United Airlines Holdings) – Un alt pilon major al zborurilor internaționale de linie.
* **JBLU** (JetBlue Airways Corp) – Operator reprezentativ pentru segmentul de cost mediu/hibrid.
* **SNCY** (Sun Country Airlines) – Operator cu o dinamică modernă, listat mai recent pe bursă.
* **RJET** (Republic Airways Holdings) – Reprezentant al aviației regionale și de transport secundar.

## Obiectiv: Detectarea Impactului Crizei Kerosenului
Pe lângă prognoza clasică a prețurilor, un pilon central și inovator al acestui proiect este evaluarea capacității rețelelor neuronale de a anticipa și detecta impactul **crizei actuale a kerosenului (combustibilul pentru avioane)** asupra acțiunilor de profil. 

Combustibilul reprezintă cea mai mare cheltuială operațională din aviație (25% - 40%). Proiectul testează dacă modelele de Deep Learning (**LSTM / GRU**) pot prezice:
1. **Schimbările bruște de regim:** Anticiparea exploziilor de volatilitate și a corecțiilor de preț declanșate de șocurile de pe piața energetică.
2. **Decuplarea performanței:** Modul în care criza afectează diferențiat giganții internaționali (`UAL`, `AAL`) față de operatorii regionali sau de nișă (`JBLU`, `SNCY`, `RJET`).

## Obiectivele Tehnice ale Predicției
**Predicție de tip continuu (Regresie):** Estimarea prețului de închidere (`Close`) pentru secvențe de 5, 10 sau 20 de zile în viitor.

## Structura Directorului
* `regresie/dataIn` — Setul de date procesat final de intrare.
* `regresie/dataOut` — Setul de date de ieșire, cu predicțiile generate de modelele LSTM ce apar dupa rularea pasilor din src, plot-uri, prognoza etc.
* `regresie/src` — Codul sursă Python ce implementează modelele LSTM, precum și funcțiile de inginerie a datelor, impartit in pasi logici.
* `regresie/model` - Directorul ce conține modelele LSTM salvate după antrenare.
* `docs/` — Documentul MS Word ce conține justificarea economică, metodologia teoretică și interpretarea rezultatelor.