# Machine Learning per l’Analisi Finanziaria  
**Laboratorio – Laurea Magistrale in Matematica**  
Università degli Studi di Napoli Federico II  

Docenti:  
- Enrico Huber  
- Pietro Soglia  

---

## 🎯 Obiettivo del corso

Il laboratorio guida gli studenti nella costruzione di un progetto completo di **churn prediction in ambito finanziario**, partendo dalla formulazione matematica del problema fino alla valutazione, interpretazione e simulazione di un mini-progetto reale.

Il percorso integra:

- Formalizzazione matematica del problema
- Analisi esplorativa dei dati (EDA)
- Preprocessing e pipeline
- Modellazione e valutazione
- Interpretabilità e implicazioni di business
- Simulazione di progetto end-to-end

---

# 📚 Struttura del corso (5 lezioni)

---

## 🔹 Lezione 1 — Il churn come problema di classificazione: formulazione e analisi esplorativa

### Obiettivo
Tradurre un problema di business in un problema matematico di classificazione.

### Parte teorica
- Il churn in ambito finanziario  
- Il problema di classificazione in Machine Learning  
- Tipologie di variabili nei dati finanziari  
- Introduzione al dataset  

### Parte pratica
- Introduzione all’uso di GenAI e Code Assistants per lo sviluppo  
- Cenni su Git e GitHub  
- Esplorazione guidata del dataset (EDA):
  - Distribuzione del target  
  - Analisi delle feature principali  
  - Verifica dell’eventuale imbalance  
  - Statistiche descrittive  
- Discussione: quali variabili potrebbero essere predittive del churn?

---

## 🔹 Lezione 2 — Dal dato grezzo al dataset modellabile

### Obiettivo
Comprendere il ruolo del preprocessing nella costruzione di un modello robusto.

### Parte teorica
- Pulizia dei dati:
  - Missing values  
  - Outlier  
- Encoding delle variabili categoriche  
- Feature scaling  
- Train / Validation / Test split  
- Il problema dell’imbalance  
- Accuracy e metriche alternative  

### Parte pratica
- Pulizia del dataset e feature engineering  
- Creazione di una pipeline base  
- Suddivisione corretta train / validation / test  

---

## 🔹 Lezione 3 — Modelli di classificazione e metriche di valutazione

### Obiettivo
Comprendere cosa significa valutare un modello in ambito finanziario.

### Parte teorica
- Modellare la realtà finanziaria  
- Overfitting e underfitting: bias-variance tradeoff  
- Modelli di classificazione:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Metriche di valutazione:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC  
  - Confusion matrix  

### Parte pratica
- Confronto tra modelli:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Calcolo e interpretazione delle metriche  
- Discussione sul tradeoff precision–recall:
  - Minimizzare falsi positivi o falsi negativi?

---

# 🏆 Churn Prediction Challenge

Attività individuale o di gruppo finalizzata allo sviluppo di un modello di churn prediction.

### Specifiche

- Obiettivo: massimizzare la metrica **ROC-AUC**
- Dataset di training fornito
- Test set non accessibile agli studenti
- Consegna: file contenente le probabilità predette
- Deadline: prima della Lezione 5
- Leaderboard anonima

La challenge simula uno scenario reale in cui il modello viene valutato su dati non osservati durante lo sviluppo.

---

## 🔹 Lezione 4 — Feature importance e interpretabilità

### Obiettivo
Collegare il modello matematico alle implicazioni di business.

### Parte teorica
- Feature importance  
- Concetto di re-training  
- Introduzione a SHAP  
- Interpretabilità nei modelli finanziari  
- Cenni regolatori e spiegabilità nel credito  

### Parte pratica
- Analisi della feature importance  
- Identificazione delle variabili più rilevanti  
- Costruzione di un semplice “profilo cliente a rischio”  

---

## 🔹 Lezione 5 — Modellazione avanzata e simulazione di progetto

### Obiettivo
Simulare un mini-progetto reale end-to-end.

### Parte teorica
- Cross-validation  
- Hyperparameter tuning (Grid Search, Random Search)  
- Introduzione concettuale a XGBoost  
- Cenni a MLOps e messa in produzione  

### Parte pratica
- Tuning di un modello  
- Selezione del modello finale  
- Costruzione di un mini-report:
  - Modello scelto  
  - Metriche  
  - Variabili rilevanti  
  - Implicazioni di business  

### Chiusura
- Discussione collettiva  
- Possibili miglioramenti  
- Next step post-produzione  

---

# 🧠 Competenze attese al termine del laboratorio

Al termine del corso lo studente sarà in grado di:

- Tradurre un problema di business in un problema di Machine Learning
- Effettuare un’analisi esplorativa strutturata
- Costruire pipeline di preprocessing corrette
- Valutare criticamente un modello di classificazione
- Interpretare i risultati in chiave finanziaria
- Simulare un flusso di progetto realistico end-to-end