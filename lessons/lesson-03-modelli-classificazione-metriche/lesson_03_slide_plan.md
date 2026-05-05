# Lesson 03 — Modelli di classificazione e metriche di valutazione

## Slide Plan

**Autori:** Enrico Huber, Pietro Soglia  
**Data prevista:** Maggio 2026  
**Formato:** 16:9 (960×540 pt) — coerente con Lezione 1 e Lezione 2  
**Numero slide target:** 36

---

## 1) Comprensione delle Lezioni 1 e 2 (dal repository)

### Lezione 1 — Il churn come problema di classificazione: EDA

- Contesto business → formalizzazione ML ($P(Y=1|X=x)$)
- Terminologia: osservazione, feature, target, modello, training/test set
- Tassonomia variabili (numeriche, categoriche, binarie)
- Data leakage e variabili non predittive (`Complain`, `RowNumber`, `CustomerId`, `Surname`)
- "Correlation ≠ causation" con esempi di correlazioni spurie
- Dataset: 10.000 righe, 18 colonne, churn rate 20.38%, 0 NaN
- Highlights EDA: Age come predittore più forte (r=0.29), Balance bimodale, differenze geografiche
- Heatmap correlazioni + "l'intruso" (`Complain` r=0.996)
- AI come copilota (policy GenAI)
- Git & GitHub
- **Live Coding** come sezione finale

### Lezione 2 — Dal dato grezzo al dataset modellabile

- Riepilogo scoperte L1 (tabella)
- Ponte "Da EDA a Preprocessing" (discovery → action)
- Pulizia: rimozione colonne non predittive + leaky
- Missing data imputation (cenni)
- Problema imbalance: paradosso DummyClassifier, Precision/Recall/F1/ROC-AUC
- ROC curve (classificatore random vs perfetto)
- Confusion matrix + trade-off Precision vs Recall (contesto business)
- Strategie imbalance: class_weight, SMOTE, undersampling, threshold tuning
- Regola aurea SMOTE (diagramma prima/dopo split)
- Outlier detection (metodo IQR)
- Feature engineering: `balance_is_zero`
- Split 60/20/20 stratificato
- Encoding: OrdinalEncoder vs OneHotEncoder
- Feature Scaling: StandardScaler vs RobustScaler vs MinMaxScaler
- ColumnTransformer Pipeline
- **Live Coding** come sezione finale

### Struttura didattica condivisa

Entrambi i deck seguono uno **scheletro identico**:

1. Cover corso → Bio docenti → Agenda corso → Cover lezione → Obiettivi → Piano sezioni
2. 4–6 sezioni tematiche con divider numerati ("01 Pulizia del dato", "02 Gestione Imbalance", ecc.)
3. Ogni sezione: slide concettuale/i → tabella/confronto → esempio/formula → insight chiave
4. Penultima sezione: **Live Coding** (singola slide divider — tutta la pratica nel notebook)
5. Chiusura: "Grazie dell'attenzione" con contatti

### Stile slide/storytelling

- **Italiano** ovunque — chiaro, professionale, gergo tecnico solo se definito
- Uso intensivo di **tabelle per confronti** (strategie, encoder, scaler, variabili)
- **Stat callout grandi** (es. "79.62% accuracy, 0% recall", "5-25x costo acquisizione")
- **Slide-citazione** con affermazioni provocatorie ("Chi abbandona lascia tracce")
- Formule con notazione matematica, sempre contestualizzate
- "Analogie" per ancorare concetti astratti (analogia esame per il leakage)
- Arco narrativo: motivazione business → teoria → evidenza dataset-specific → azione

### Stile visivo (dai PDF)

- **Formato:** 16:9 (960×540 pt)
- **Densità:** Slide contenuto 200–700 caratteri; tabelle fino a ~1000; slide con grafici 20–100 caratteri
- **Colori:** Divider sezione scuri (numeri grandi + titolo), slide contenuto bianche/chiare
- **Branding:** "Here to Dare." sui cover, "IL CHURN COME PROBLEMA DI CLASSIFICAZIONE" come header persistente
- **Tipi di slide osservati:**
  - Cover (branded, testo minimale)
  - Bio slide (foto + ruoli)
  - Agenda (5 lezioni, corrente evidenziata)
  - Cover lezione (titolo + data)
  - Obiettivi (lista numerata, 4–8 punti)
  - Rollout/Plan (lista numerata visiva delle sezioni)
  - Divider sezione (sfondo scuro, numero, titolo)
  - Spiegazione concetto (bullet + paragrafo breve)
  - Slide tabella/confronto
  - Slide formula (stile LaTeX + testo contestuale)
  - Stat callout (numero grande + spiegazione)
  - Grafico/plot (immagine embedded dal notebook)
  - Diagramma (flusso processo, es. SMOTE prima/dopo)
  - Slide citazione (testo grande corsivo)
  - Divider live coding
  - Chiusura/contatti

### Pattern da preservare nella Lezione 3

- Recap della lezione precedente come ponte di apertura (formato tabella)
- Divider sezione numerati per il pacing
- Una "citazione provocatoria" per sezione principale
- Tabelle per confronti modelli/strategie
- Numeri concreti dal dataset in ogni slide concettuale
- Live coding come ultima sezione logica
- ~36 slide totali — densità sufficiente per una sessione di 2–3 ore

### Punti deboli / opportunità di miglioramento

- L2 ha slide con troppo testo (pagina 10: 1149 char, pagina 22: 1028 char) — meglio dividere
- La sezione "Live coding" è un singolo divider senza preview dei contenuti — aggiungere una slide "cosa coderemo" migliorerebbe la transizione
- Non ci sono slide "checkpoint/domanda interattiva" — L3 potrebbe introdurre 1–2 pause
- I grafici del notebook non sono contestualizzati sulle slide (appaiono standalone) — aggiungere brevi annotazioni sulle slide con grafici migliorerebbe la comprensione

---

## 2) Comprensione dei materiali Lezione 3 (dal repository)

### Contenuto esistente

- **`notebooks/lesson-03/`**: Vuoto (solo `.gitkeep`)
- **`outputs/models/`**: Vuoto — nessun modello addestrato
- **`outputs/predictions/`**: Vuoto — nessuna predizione salvata
- **Artefatti Lezione 2 pronti per il consumo:**
  - `outputs/data/lesson_02_X_train.parquet` (6.000 × 30)
  - `outputs/data/lesson_02_X_val.parquet` (2.000 × 30)
  - `outputs/data/lesson_02_X_test.parquet` (2.000 × 30)
  - `outputs/data/lesson_02_y_train.parquet`
  - `outputs/data/lesson_02_y_val.parquet`
  - `outputs/data/lesson_02_y_test.parquet`
  - `outputs/data/lesson_02_preprocessor.pkl`
  - `outputs/data/lesson_02_feature_names.json` (30 nomi feature)

### Concetti esplicitamente promessi per la Lezione 3

**Dal notebook L2 "Riepilogo":**
> "Nella Lezione 3 utilizzeremo il dataset modellabile salvato per:
> - Costruire e confrontare modelli: Logistic Regression, Decision Tree, Random Forest
> - Calcolare e interpretare le metriche: Accuracy, Precision, Recall, F1, ROC-AUC
> - Visualizzare la Confusion Matrix e la ROC Curve
> - Discutere il trade-off Precision–Recall nel contesto del churn bancario"

**Dal README:**
> "3. Modelli di classificazione e metriche di valutazione: confronto modelli e valutazione"

**Dall'agenda del corso (slide 3):**
> "Modelli di classificazione e metriche di valutazione"

**Dalla L1 (slide 13 — terminologia):**
> "In Lezione 3 vedremo modelli di classificazione come Logistic Regression, Decision Tree, Random Forest"

**Dalla L2 (slide 21):**
> "La soglia di default è 0.5, ma verrà scelta analizzando il Validation Set (Lezione 3)"

### Materiali riutilizzabili

- **Baseline dalla L1:** LogReg ROC-AUC ≈ 0.85, recall = 37% (151/408 churner identificati)
- **Grafico esistente:** `outputs/figures/lesson_01_confusion_matrix_baseline.png`
- **Grafico esistente:** `outputs/figures/lesson_01_roc_curve_baseline.png`
- **Metriche già introdotte:** Precision, Recall, F1, ROC-AUC (L2 slide 16–21)
- **Strategie imbalance spiegate:** class_weight, SMOTE, threshold tuning (L2)
- **Dati preprocessati pronti:** file Parquet con 30 feature

### Cosa manca o va creato

- Il notebook della Lezione 3 non esiste ancora — le slide guideranno la creazione
- I modelli (LogReg, DecisionTree, RandomForest) vanno addestrati → produrranno confusion matrix, ROC curve, tabelle metriche
- L'ottimizzazione della soglia sul validation set va dimostrata
- La tabella di confronto modelli (tutti side by side) va creata
- La visualizzazione del decision tree va prodotta
- Opzionalmente: risultati cross-validation, learning curve

### Come la Lezione 3 segue naturalmente dalle Lezioni 1 e 2

- **L1** ha identificato il problema, esplorato i dati, prodotto una baseline rapida (LogReg AUC=0.85, recall=37%)
- **L2** ha pulito i dati, costruito una pipeline corretta, salvato un dataset "modeling-ready"
- **L3** è il payoff: finalmente addestriamo modelli veri su dati preparati correttamente e li valutiamo rigorosamente — rispondendo a "il nostro preprocessing ha davvero migliorato le cose?"

---

## 3) Piano slide proposto per la Lezione 3

| Slide # | Titolo proposto | Obiettivo | Messaggio chiave | Idea visiva/contenuto | Materiale sorgente | Flusso speaker / Nota transizione | Interazione opzionale |
|---------|----------------|-----------|------------------|-----------------------|-------------------|----------------------------------|----------------------|
| 1 | **Cover Corso** — "Here to Dare. Machine Learning per l'analisi finanziaria" | Continuità brand | Apertura standard | Stesso cover branded di L1/L2, "Maggio \| 2026" | Template L1/L2 slide 1 | — | — |
| 2 | **Chi siamo** | Intro docenti | — | Enrico Huber + Pietro Soglia, stesso layout | Template L1/L2 slide 2 | Saluto breve, skip se stesso pubblico | — |
| 3 | **Agenda del corso** | Posizionare la lezione nel curriculum | "Oggi siamo qui: Lezione 3" | 5 lezioni, **#3 evidenziata** | Template L1/L2 slide 3 | "Siamo a metà del percorso" | — |
| 4 | **Cover Lezione** — "Modelli di classificazione e metriche di valutazione" | Title card lezione | — | Titolo + data "4 Maggio 2026" + "Here to Dare." + "Public" | Template L1/L2 slide 4 | — | — |
| 5 | **Obiettivi della lezione** | Definire aspettative | 6 obiettivi di apprendimento | Lista numerata: 1) Caricare dataset modellabile 2) Addestrare LogReg/DT/RF 3) Calcolare metriche 4) Visualizzare Confusion Matrix e ROC 5) Confrontare modelli 6) Ottimizzare soglia di classificazione | Agenda corso + promesse L2 | "Oggi passiamo dalla preparazione all'azione" | — |
| 6 | **Programma della lezione** (Rollout Plan) | Roadmap visiva sezioni | 6 sezioni numerate | 01 RIEPILOGO E CARICAMENTO DATI / 02 MODELLI DI CLASSIFICAZIONE / 03 METRICHE DI VALUTAZIONE / 04 CONFRONTO MODELLI / 05 THRESHOLD TUNING / 06 LIVE CODING | Pattern da L1 p6, L2 p6 | "Ecco la roadmap di oggi" | — |
| 7 | **[DIVIDER 01]** — "Riepilogo e caricamento dati" | Transizione alla prima sezione | — | Sfondo scuro, numero "01", titolo sezione | Pattern da L1 p7, L2 p9 | — | — |
| 8 | **Dove eravamo** — Riepilogo Lezione 2 | Ponte dalla lezione precedente | "Il dataset è pronto; oggi modelliamo" | Tabella: Operazione / Risultato (stesso formato di L2 p7) — pipeline, split, conteggio feature | Notebook L2 sezione "Riepilogo" | "Nella lezione scorsa abbiamo costruito la pipeline…" | "Qualcuno ricorda quante feature ha il dataset processato?" |
| 9 | **Il nostro punto di partenza** | Chiarire gli input di oggi | "30 feature, 6.000 train, 2.000 val, 2.000 test" | Stat callout grande: "30 feature × 6.000 esempi" + file elencati | `lesson_02_feature_names.json`, file Parquet | "Oggi partiamo direttamente dal dataset modellabile salvato in Lezione 2" | — |
| 10 | **Dalla baseline alla modellazione** | Motivare il miglioramento | "Il baseline va superato" | Tabella: Baseline L1 vs obiettivo L3. Mostrare LogReg (no pipeline) AUC=0.85, recall=37% → target: pipeline corretta + modelli migliori | Risultati baseline L1 | "Il 37% di recall non è accettabile per un sistema di retention. Possiamo fare meglio?" | "Secondo voi, cosa ha limitato il baseline?" |
| 11 | **[DIVIDER 02]** — "Modelli di classificazione" | Transizione alla teoria modelli | — | Sfondo scuro, "02", "Modelli di classificazione" | Pattern | — | — |
| 12 | **"Tre famiglie, un obiettivo"** | Panoramica dei 3 modelli | Approcci diversi, stesso goal: stimare P(Y=1\|X) | Slide citazione: "Tre algoritmi, tre filosofie: lineare, ad albero, ad ensemble. Stesso obiettivo: separare churner da non-churner." | Agenda corso | "Oggi confronteremo tre modelli con complessità crescente" | — |
| 13 | **Logistic Regression** — come funziona | Intuizione per LogReg | Linea di separazione nello spazio delle feature | Sinistra: formula $\sigma(w^T x + b) = \frac{1}{1+e^{-(w^T x + b)}}$. Destra: curva sigmoide + decision boundary | Teoria ML standard | "Il modello più semplice: una combinazione lineare delle feature, passata per una sigmoide" | — |
| 14 | **Logistic Regression** — pro e contro | Quando usarla | Semplice, interpretabile, baseline solida | Tabella: Pro (interpretabile, veloce, probabilità calibrate) / Contro (solo boundary lineare, non cattura interazioni) / Quando (primo modello, baseline, feature engineering fatto bene) | Teoria ML standard | "Se la relazione è lineare, LogReg basta. Altrimenti..." | — |
| 15 | **Decision Tree** — come funziona | Intuizione per DT | Sequenza di domande binarie sui dati | Diagramma: albero con nodi "Age > 42?", "IsActive = 1?", foglia "Churn=1 (prob 0.68)" | Teoria ML + feature del dataset | "Un albero decisionale fa domande sequenziali: 'il cliente ha più di 42 anni?' 'è attivo?'" | "Qual è il vantaggio rispetto a LogReg?" |
| 16 | **Decision Tree** — pro e contro | Punti di forza e debolezza | Non-linearità + interpretabilità, ma overfitting | Tabella: Pro (cattura non-linearità, interpretabile, no scaling necessario) / Contro (overfitting, instabile, alta varianza) / Quando (esplorazione, interpretabilità richiesta) | Teoria ML standard | "L'albero è potente ma fragile: un singolo split diverso cambia tutto" | — |
| 17 | **Random Forest** — come funziona | Intuizione per RF | Molti alberi votano insieme (wisdom of the crowd) | Diagramma: alberi multipli → voto maggioranza → predizione. "Bagging: campionamento con rimpiazzamento" | Teoria ML standard | "Se un albero è instabile, molti alberi sono robusti. Questa è l'idea del Random Forest" | — |
| 18 | **Random Forest** — pro e contro | Quando usarlo | Robusto, preciso, meno interpretabile | Tabella: Pro (robusto a overfitting, cattura interazioni, out-of-bag error) / Contro (meno interpretabile, più lento, molti iperparametri) / Quando (accuracy è prioritaria, dataset di dimensioni medie) | Teoria ML standard | "Il tradeoff: guadagniamo accuratezza, perdiamo trasparenza" | — |
| 19 | **Tabella comparativa dei 3 modelli** | Modello mentale side-by-side | Ogni modello ha il suo spazio | Tabella: Modello / Tipo boundary / Gestisce non-linearità / Interpretabilità / Rischio overfitting / Speed | Sintesi | "Teniamo questa tabella in mente mentre vediamo i risultati" | — |
| 20 | **class_weight="balanced"** — il nostro approccio | Ricordare strategia imbalance | Riutilizziamo la strategia vista in L2 | Recap breve: formula peso + "Tutti i modelli oggi useranno class_weight='balanced'" | L2 slide 23 | "Ricordate da Lezione 2? Ogni errore su un churner pesa 2.45×" | — |
| 21 | **[DIVIDER 03]** — "Metriche di valutazione" | Transizione alla valutazione | — | Sfondo scuro, "03", "Metriche di valutazione" | Pattern | — | — |
| 22 | **"Un modello senza metriche è un'opinione"** | Motivare valutazione rigorosa | La valutazione distingue ML dal guesswork | Slide citazione | Originale | "Senza numeri, non sappiamo se abbiamo migliorato qualcosa" | — |
| 23 | **Confusion Matrix** — anatomia | Decodificare la matrice | TP, TN, FP, FN nel contesto churn | Diagramma matrice 2×2 con: TP="churner identificato (salvato!)", FP="contatto inutile (costo)", FN="churner perso (danno)", TN="corretto non-intervento" | L2 slide 20 + contesto business | "Ogni cella ha un costo business diverso" | "Quale cella è più costosa per la banca?" |
| 24 | **Metriche derivate dalla matrice** | Reference rapido formule | Precision, Recall, F1, Accuracy — riepilogo visivo | Tabella (compatta, stile reference): come L2 slide 16 ma più compatta. Aggiungere: "Oggi calcoleremo tutte queste per i nostri 3 modelli" | Recap L2 slide 16 | "Queste le conosciamo dalla Lezione 2. Oggi le calcoleremo davvero." | — |
| 25 | **ROC Curve e AUC** — riepilogo | Reference per interpretazione | Come leggere la curva in pratica | Diagramma ROC curve (reference + annotazione: "più in alto a sinistra = meglio") | L2 slide 17–19, chart ROC L1 | "La ROC ci dà una misura globale: quanto bene il modello separa le classi?" | — |
| 26 | **[DIVIDER 04]** — "Confronto modelli" | Transizione ai risultati | — | Sfondo scuro, "04", "Confronto modelli" | Pattern | — | — |
| 27 | **Risultati: tabella comparativa** | Mostrare lo scoreboard | Numeri reali dai nostri esperimenti | Tabella: Modello / AUC / Precision / Recall / F1 / Accuracy — per LogReg, DT, RF (valori dal notebook). Evidenziare best-in-class per metrica | Da generare dal notebook | "Ecco i numeri. Nessun modello domina su tutte le metriche." | "Quale modello scegliereste e perché?" |
| 28 | **Confusion Matrix a confronto** | Confronto visivo | Dove sbagliano i modelli | 3 confusion matrix side by side (heatmap dal notebook) | Da generare dal notebook | "Guardate gli FN: quanti churner perde ciascun modello?" | — |
| 29 | **ROC Curve: tre modelli a confronto** | Ranking visivo modelli | AUC come classifica globale | Singolo plot con 3 ROC curve sovrapposte + diagonale reference | Da generare dal notebook | "Random Forest domina, ma di quanto? E a quale costo computazionale?" | — |
| 30 | **"Il modello migliore dipende dalla domanda"** | Sfumatura: nessun vincitore universale | Trade-off business rivisitato | 2 scenari: "Budget limitato" → alta Precision (RF) vs "Nessun churner deve sfuggire" → alta Recall (LogReg con soglia bassa) | L2 slide 21 estesa | "Non esiste il modello migliore in assoluto. Esiste il modello migliore per il vostro obiettivo." | — |
| 31 | **[DIVIDER 05]** — "Threshold tuning" | Transizione al threshold | — | Sfondo scuro, "05", "Threshold tuning" | Pattern | — | — |
| 32 | **Soglia di default vs soglia ottimale** | Spiegare il concetto di soglia | 0.5 non è sempre la scelta giusta | Sinistra: "Soglia = 0.5 → Recall basso". Destra: "Soglia = 0.3 → Recall alto, Precision cala". Diagramma: metafora slider | Promessa L2 slide 21 | "In Lezione 2 abbiamo detto: la soglia si ottimizza sul validation set. Oggi lo facciamo." | — |
| 33 | **Precision-Recall trade-off** — curva sul nostro dataset | Mostrare curva PR | Come si muovono Precision e Recall al variare della soglia | Curva Precision-Recall (dal notebook) con annotazione "punto operativo scelto" | Da generare dal notebook | "Muovendo la soglia, guadagniamo recall ma perdiamo precision. Dove ci fermiamo?" | "Se la banca vuole recall ≥ 60%, quale precision otteniamo?" |
| 34 | **Risultato: modello finale con soglia ottimizzata** | Risultato conclusivo | Il nostro "best model" | Stat grandi: "Random Forest + soglia 0.35 → Recall = 65%, Precision = 52%, AUC = 0.88" (valori placeholder — reali dal notebook) | Da generare dal notebook | "Questo è il nostro modello per la Lezione 4, dove ne studieremo l'interpretabilità" | — |
| 35 | **[DIVIDER 06]** — "Live coding" | Transizione alla pratica | — | Sfondo scuro, "06", "Live coding" | Pattern da L1/L2 | "Apriamo il notebook e mettiamo in pratica tutto" | — |
| 36 | **Grazie dell'attenzione** | Chiusura | Contatti | Stesso layout: Enrico + Pietro contatti | Template L1/L2 slide 36 | — | — |

---

## 4) Allineamento stilistico con i PDF delle Lezioni 1 e 2

### Pattern visivi da riutilizzare

- **Prime 6 slide identiche:** Cover → Bio → Agenda → Cover lezione → Obiettivi → Rollout Plan. Questo scheletro è non negoziabile.
- **Divider sezione:** Sfondo scuro, numero grande "0N", titolo sezione, header "IL CHURN COME PROBLEMA DI CLASSIFICAZIONE". Stessa logica font e colori.
- **Tabelle di confronto:** Dispositivo didattico dominante in entrambi i deck. La L3 le usa per: confronto modelli, risultati metriche, scenari business.
- **Stat callout grandi:** Stile "79.62% accuracy, 0% recall". La L3 li usa per i numeri di performance del modello finale.
- **Slide citazione:** Pattern "Chi abbandona lascia tracce". La L3 ha 2–3 ancore provocatorie ("Un modello senza metriche è un'opinione", "Tre famiglie, un obiettivo", "Il modello migliore dipende dalla domanda").
- **Slide formula:** Concise, con spiegazione contestuale accanto alla matematica. Usate con parsimonia (sigmoide per LogReg, Gini impurity per DT — se necessario).
- **Divider "Live coding" come penultima slide**, sempre.

### Tipi di slide da replicare

- Tabella recap (come L2 p7)
- Concetto + tabella (come L2 p22 tabella strategie)
- Formula + diagramma (come L2 p26 IQR)
- Inserimento grafico (come L1 p29–32 grafici EDA)
- Diagramma di processo (come L2 p24 SMOTE prima/dopo)

### Linee guida densità

- **Target:** 200–600 caratteri per slide contenuto; max 800 per tabelle
- **Regola pratica:** Se servono >800 caratteri, dividere in 2 slide
- **Slide con grafico:** Titolo + annotazione di 1 frase sola (20–100 caratteri)
- **Tabelle:** Max 5 colonne, max 6 righe (leggibili in proiezione 16:9)

### Grafici/diagrammi/screenshot da usare

- Confusion matrix (stile heatmap, da seaborn/sklearn)
- ROC curve (sovrapposte, da sklearn)
- Curva Precision-Recall
- Visualizzazione decision tree (max depth 3 per leggibilità)
- Bar chart confronto metriche modelli (metriche side by side)
- Opzionalmente: diagramma sigmoide per intuizione LogReg
- Opzionalmente: sketch concettuale "foresta di alberi" per RF

### Cosa evitare

- Muri di testo (>5 bullet point per slide)
- Snippet di codice sulle slide (il codice vive nel notebook)
- Diagrammi generici non ancorati al dataset reale
- Introdurre nuovo preprocessing o EDA (quello era L1–L2)
- Formalismo matematico eccessivo (restare intuitivi, le formule supportano)
- Più di 36 slide (mantenere lunghezza deck coerente)

### Far sentire la Lezione 3 come continuazione naturale

- La lezione evidenziata nell'agenda (slide 3) si sposta su #3
- La tabella recap (slide 8) referenzia esplicitamente gli artefatti L2 per nome
- Il vocabolario delle metriche è stato introdotto in L2 → L3 lo applica (nessuna ri-spiegazione, solo slide reference rapida)
- La promessa "La soglia si sceglie sul Validation Set (Lezione 3)" dalla L2 slide 21 viene esplicitamente mantenuta
- La baseline dalla L1 (AUC=0.85, recall=37%) è il benchmark da superare

---

## 5) Migliori materiali da riutilizzare

| File/Artefatto | Come usare nelle slide |
|----------------|----------------------|
| `outputs/data/lesson_02_X_train.parquet` | Reference su slide "punto di partenza" (6.000 × 30) |
| `outputs/data/lesson_02_feature_names.json` | Mostrare 30 nomi feature se serve contesto |
| `outputs/data/lesson_02_preprocessor.pkl` | Menzionare come "pipeline pronta" nel recap |
| `outputs/figures/lesson_01_confusion_matrix_baseline.png` | Mostrare come "baseline L1" prima dei nuovi risultati |
| `outputs/figures/lesson_01_roc_curve_baseline.png` | Punto di riferimento per confronto ROC |
| Notebook L2 — sezione "Riepilogo" | Sorgente per contenuto tabella recap |
| Notebook L2 — calcolo class_weight | Reference breve per reminder imbalance |
| L2 slide 16 (formule metriche) | Versione compatta reference per slide "metriche derivate" |
| L2 slide 21 (tabella trade-off) | Estendere per slide scenario business |
| L2 slide 22 (tabella strategie) | Richiamo breve della strategia scelta |
| L1 slide 11 (diagramma business → ML) | Callback possibile a "oggi completiamo la pipeline" |
| L1 slide 13 (terminologia) | Promessa mantenuta: "In Lezione 3 vedremo..." |

### Output notebook generati (verificati dopo ri-esecuzione)

- `outputs/figures/lesson_03_confusion_matrices.png` ✓
- `outputs/figures/lesson_03_roc_curve_comparison.png` ✓
- `outputs/figures/lesson_03_pr_curve.png` ✓
- `outputs/figures/lesson_03_dt_depth_curve.png` ✓
- `outputs/figures/lesson_03_rf_feature_importance.png` ✓
- `outputs/data/lesson_03_best_model.pkl` ✓ (RF finale, ~31 MB)
- `outputs/data/lesson_03_metrics.json` ✓ (metriche train/val/test)

---

## 6) Gap e raccomandazioni

### Slide aggiuntive che potrebbero aiutare

- **"Cos'è un iperparametro?"** (1 slide) — distinzione breve parametro vs iperparametro, preparando L4/L5 dove si farà tuning. Tenerla a 3 righe di definizione, non un approfondimento.
- **"Overfitting vs Underfitting"** (1 slide) — visual con performance training/val che divergono. Motiva perché DT da solo va in overfitting e RF no. È implicito ma mai insegnato esplicitamente in L1/L2.
- **"Come leggere i risultati"** (1 slide prima della tabella comparativa) — micro-guida: "guardate prima AUC per il ranking, poi recall per il business, poi precision per il budget"

### Cosa creare dagli output del notebook

- Tutte le heatmap delle confusion matrix
- ROC curve sovrapposte (3 modelli + diagonale)
- Curva Precision-Recall per il modello migliore
- Visualizzazione decision tree (depth=3, con nomi feature)
- Opzionale: bar chart per confronto metriche tra modelli

### Cosa semplificare, tagliare, unire o espandere

- **Tagliare:** Ri-spiegare ROC da zero (già fatto in L2). Solo 1 slide reference.
- **Unire:** Pro/contro LogReg potrebbe unirsi a "come funziona" se lo spazio è stretto.
- **Espandere:** La sezione threshold tuning merita 2–3 slide (è stata molto anticipata in L2 ma mai mostrata).
- **Semplificare:** Non spiegare Gini impurity in dettaglio — menzionare che esiste. Focus sull'intuizione.

### Ponti narrativi dalla Lezione 2

- L'apertura (slide 8–10) dice esplicitamente: "In Lezione 2 abbiamo costruito la pipeline e salvato i dati. Oggi li usiamo."
- Il reminder `class_weight='balanced'` (slide 20) crea continuità con la sezione imbalance di L2.
- Il threshold tuning (slide 32–34) mantiene esplicitamente la promessa fatta nella L2 slide 21.
- La chiusura dovrebbe anticipare la Lezione 4: "Abbiamo il modello migliore. Ma *perché* predice churn? Lo vedremo nella Lezione 4 con la feature importance."

---

## 7) Versione finale raccomandata

### Sequenza finale raccomandata (36 slide)

1. Cover corso — "Here to Dare. Machine Learning per l'analisi finanziaria — Maggio | 2026"
2. Chi siamo — Enrico Huber + Pietro Soglia
3. Agenda del corso — 5 lezioni, **Lezione 3 evidenziata**
4. Cover lezione — "Modelli di classificazione e metriche di valutazione" + data
5. Obiettivi della lezione — 6 punti numerati
6. Programma della lezione — Rollout Plan (6 sezioni numerate)
7. **[DIVIDER 01]** — Riepilogo e caricamento dati
8. Dove eravamo — tabella riepilogo L2
9. Il nostro punto di partenza — stat callout: 30 feature, 6.000 train, 2.000 val, 2.000 test
10. Dalla baseline alla modellazione — baseline L1 (AUC=0.85, recall=37%) come benchmark
11. **[DIVIDER 02]** — Modelli di classificazione
12. "Tre famiglie, un obiettivo" — slide citazione
13. Logistic Regression — intuizione (sigmoide + boundary lineare)
14. Logistic Regression — pro/contro (tabella)
15. Decision Tree — intuizione (albero con split su feature del dataset)
16. Decision Tree — pro/contro (tabella)
17. Random Forest — intuizione (ensemble di alberi → voto di maggioranza)
18. Random Forest — pro/contro (tabella)
19. Tabella comparativa dei 3 modelli (boundary / non-linearità / interpretabilità / overfitting / speed)
20. class_weight="balanced" — reminder (breve, 1 slide)
21. **[DIVIDER 03]** — Metriche di valutazione
22. "Un modello senza metriche è un'opinione" — slide citazione
23. Confusion Matrix — anatomia nel contesto churn (costi business per cella)
24. Metriche derivate — tabella compatta (Precision, Recall, F1, AUC)
25. ROC Curve e AUC — diagramma reference con annotazione
26. **[DIVIDER 04]** — Confronto modelli
27. Risultati: tabella comparativa (3 modelli × 5 metriche, da notebook)
28. Confusion Matrix a confronto (3 heatmap side-by-side, da notebook)
29. ROC Curve: tre modelli sovrapposti (chart da notebook)
30. "Il modello migliore dipende dalla domanda" — 2 scenari business
31. **[DIVIDER 05]** — Threshold tuning
32. Soglia di default vs soglia ottimale — metafora slider + concetto
33. Curva Precision-Recall sul nostro dataset (chart + punto operativo)
34. Risultato: modello finale con soglia ottimizzata — stat callout grande
35. **[DIVIDER 06]** — Live coding
36. Grazie dell'attenzione — contatti

---

## 8) Contenuto testuale per ogni slide

Di seguito il testo effettivo da inserire in ogni slide, pronto per il copia-incolla nel tool di presentazione. I valori numerici provengono dal notebook `lesson_03.py` eseguito (file `outputs/data/lesson_03_metrics.json`).

---

### Slide 1 — Cover Corso

**Testo principale:**
> HERE TO DARE.

**Sottotitolo:**
> Machine Learning per l'Analisi Finanziaria

**Footer:**
> Maggio | 2026 — Public

---

### Slide 2 — Chi siamo

| | |
|--|--|
| **Enrico Huber** | **Pietro Soglia** |
| Data Scientist | Data Scientist |
| enrico.huber@... | pietro.soglia@... |

*(Stesso layout slide 2 di L1/L2 — foto + ruoli + contatti)*

---

### Slide 3 — Agenda del corso

| # | Lezione | Stato |
|---|---------|-------|
| 1 | Il churn come problema di classificazione: EDA | ✓ |
| 2 | Dal dato grezzo al dataset modellabile | ✓ |
| **3** | **Modelli di classificazione e metriche di valutazione** | **← OGGI** |
| 4 | Interpretabilità e spiegabilità del modello | |
| 5 | Tuning, validazione e deployment | |

---

### Slide 4 — Cover Lezione

**Titolo:**
> MODELLI DI CLASSIFICAZIONE E METRICHE DI VALUTAZIONE

**Sottotitolo:**
> Lezione 03

**Footer:**
> 4 Maggio 2026 — Here to Dare. — Public

---

### Slide 5 — Obiettivi della lezione

**Titolo:** Obiettivi della lezione

1. Caricare il dataset modellabile prodotto dalla Lezione 2.
2. Addestrare e confrontare **Logistic Regression, Decision Tree e Random Forest**.
3. Comprendere empiricamente il **bias-variance tradeoff** attraverso la curva di profondità.
4. Calcolare e interpretare **Accuracy, Precision, Recall, F1 e ROC-AUC**.
5. Visualizzare e confrontare **Confusion Matrix** e **ROC Curve** per ogni modello.
6. Ottimizzare la **soglia di classificazione** sul validation set.

---

### Slide 6 — Programma della lezione (Rollout Plan)

**Titolo:** Programma

| # | Sezione |
|---|---------|
| 01 | RIEPILOGO E CARICAMENTO DATI |
| 02 | MODELLI DI CLASSIFICAZIONE |
| 03 | METRICHE DI VALUTAZIONE |
| 04 | CONFRONTO MODELLI |
| 05 | THRESHOLD TUNING |
| 06 | LIVE CODING |

---

### Slide 7 — [DIVIDER 01]

**Sfondo scuro**

> **01**
>
> RIEPILOGO E CARICAMENTO DATI

---

### Slide 8 — Dove eravamo

**Titolo:** Riepilogo Lezione 2

| Operazione | Risultato |
|------------|-----------|
| Rimozione variabili non predittive | `RowNumber`, `CustomerId`, `Surname` eliminati |
| Rimozione leakage | `Complain` (r = 0.996 con target) eliminato |
| Feature engineering | `balance_is_zero` creata dalla bimodalità di Balance |
| Encoding | OneHotEncoder per categoriche → 24 colonne |
| Scaling | StandardScaler per 6 feature numeriche |
| Split stratificato | 60% train / 20% val / 20% test |
| Pipeline salvata | `lesson_02_preprocessor.pkl` |

**Nota speaker:** "Nella lezione scorsa abbiamo costruito una pipeline riproducibile e salvato il dataset pronto per la modellazione."

---

### Slide 9 — Il nostro punto di partenza

**Titolo:** Il dataset modellabile

**Stat callout grande:**

> **30 feature × 6.000 esempi**

**Dettaglio:**

| Split | Dimensione | Churn rate |
|-------|-----------|------------|
| Train | 6.000 × 30 | 20.38% |
| Validation | 2.000 × 30 | 20.40% |
| Test | 2.000 × 30 | 20.35% |

**Nota:** 6 numeriche scalate + 24 categoriche one-hot encoded. Zero NaN. Stratificazione confermata.

---

### Slide 10 — Dalla baseline alla modellazione

**Titolo:** Il baseline da superare

| | Baseline L1 (LogReg senza pipeline) | Obiettivo L3 |
|--|------|------|
| ROC-AUC | 0.85 | > 0.85 |
| Recall churn | 37% (151/408) | > 60% |
| Pipeline | Fit su tutto il dataset (leakage!) | Fit solo su train |

**Citazione:**
> "Il 37% di recall non è accettabile per un sistema di retention. Oggi faremo meglio — e in modo corretto."

**Nota speaker:** "La baseline della Lezione 1 era veloce ma viziata: lo scaler era fittato sull'intero dataset. Oggi partiamo da una pipeline pulita."

---

### Slide 11 — [DIVIDER 02]

**Sfondo scuro**

> **02**
>
> MODELLI DI CLASSIFICAZIONE

---

### Slide 12 — "Tre famiglie, un obiettivo"

**Slide citazione (testo grande, centrato):**

> "Tre algoritmi, tre filosofie: lineare, ad albero, ad ensemble.
> Stesso obiettivo: stimare $P(\text{Churn}=1 \mid X)$."

**Nota speaker:** "Oggi confronteremo tre modelli con complessità crescente. Ognuno ha i suoi punti di forza — li scopriremo dai dati."

---

### Slide 13 — Logistic Regression — come funziona

**Titolo:** Logistic Regression — Intuizione

**Formula (centrata):**

$$\hat{p} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Spiegazione (a destra della formula):**
- Combinazione lineare delle feature → singolo numero
- La **sigmoide** $\sigma$ comprime il risultato tra 0 e 1 → probabilità
- Il **decision boundary** è un iperpiano nello spazio delle feature

**Diagramma:** Curva sigmoide con annotazione "Se $w^T x + b > 0$ → $\hat{p} > 0.5$ → prediciamo Churn"

**Nota speaker:** "Il modello più semplice: una combinazione lineare delle feature, trasformata in probabilità dalla sigmoide."

---

### Slide 14 — Logistic Regression — pro e contro

**Titolo:** Logistic Regression — Quando usarla

| Pro ✓ | Contro ✗ |
|-------|----------|
| Interpretabile (un peso per feature) | Solo decision boundary lineare |
| Veloce da addestrare (secondi) | Non cattura interazioni tra feature |
| Probabilità calibrate nativamente | Sensibile a feature non scalate |
| Ottima baseline di riferimento | Sottoperforma se relazioni non-lineari |

**Quando usarla:**
- Come primo modello (baseline)
- Quando l'interpretabilità è prioritaria
- Quando le feature sono ben ingegnerizzate (feature engineering compensa la linearità)

---

### Slide 15 — Decision Tree — come funziona

**Titolo:** Decision Tree — Intuizione

**Diagramma albero (3 livelli):**

```
         Age > 42?
        /         \
      Sì           No
      /               \
  NumOfProducts > 2?   IsActive = 0?
   /        \            /        \
 Churn     Balance>0?  Churn    Non-Churn
(p=0.72)    ...       (p=0.35)  (p=0.12)
```

**Testo:** L'albero impara una sequenza di **domande binarie** (split) che partizionano lo spazio delle feature. Ogni foglia produce una probabilità di churn.

**Nota speaker:** "Un albero decisionale ragiona come un analista: 'il cliente ha più di 42 anni? Ha più di 2 prodotti?' Sequenza di domande, fino a una decisione."

---

### Slide 16 — Decision Tree — pro e contro

**Titolo:** Decision Tree — Quando usarlo

| Pro ✓ | Contro ✗ |
|-------|----------|
| Cattura non-linearità e interazioni | **Overfitting** senza vincoli di profondità |
| Interpretabile (si disegna l'albero) | Instabile: un dato diverso → albero diverso |
| Non richiede feature scaling | Alta varianza (piccole perturbazioni → grandi cambiamenti) |
| Feature importance naturale | Prestazioni spesso inferiori agli ensemble |

**Stat dal notebook:**
> DT senza limiti: Train AUC = **1.000**, Val AUC = **0.663** → Overfitting totale

**Nota speaker:** "L'albero è potente ma fragile. Senza limiti, memorizza ogni singolo esempio del training set — e perde ogni capacità predittiva su dati nuovi."

---

### Slide 17 — Random Forest — come funziona

**Titolo:** Random Forest — Intuizione

**Diagramma:**
```
Training set
    ↓ (Bootstrap: campionamento con rimpiazzamento)
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Albero 1   │  │  Albero 2   │  │  Albero N   │
│ (subset     │  │ (subset     │  │ (subset     │
│  feature)   │  │  feature)   │  │  feature)   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       ↓                ↓                ↓
       Churn           Non-Churn         Churn
                    ↓
            VOTO DI MAGGIORANZA → Churn (2/3)
```

**Testo:** Molti alberi diversificati votano insieme. La **decorrelazione** (bootstrap + subset di feature) riduce la varianza senza aumentare il bias.

**Nota speaker:** "Se un albero è instabile, molti alberi sono robusti. Questa è la filosofia del Random Forest: wisdom of the crowd."

---

### Slide 18 — Random Forest — pro e contro

**Titolo:** Random Forest — Quando usarlo

| Pro ✓ | Contro ✗ |
|-------|----------|
| Robusto a overfitting (bagging) | Meno interpretabile (200 alberi) |
| Cattura interazioni complesse | Più lento da addestrare |
| Out-of-bag error come stima interna | Molti iperparametri (n_estimators, max_depth, ...) |
| Feature importance integrata | Richiede più memoria |

**Stat dal notebook:**
> RF (200 alberi): Val ROC-AUC = **0.872** — il migliore osservato

**Nota speaker:** "Il tradeoff: guadagniamo capacità predittiva, perdiamo trasparenza. Ma l'AUC parla chiaro."

---

### Slide 19 — Tabella comparativa dei 3 modelli

**Titolo:** Tre modelli a confronto — caratteristiche

| Caratteristica | Logistic Regression | Decision Tree | Random Forest |
|---------------|-------------------|---------------|---------------|
| Tipo boundary | Lineare (iperpiano) | Non-lineare (assi paralleli) | Non-lineare (ensemble) |
| Gestisce non-linearità | ✗ (solo con feature eng.) | ✓ | ✓ |
| Interpretabilità | Alta (pesi) | Media (albero visualizzabile) | Bassa (black-box) |
| Rischio overfitting | Basso | **Alto** (senza vincoli) | Medio (bagging regolarizza) |
| Velocità training | ⚡ Molto veloce | ⚡ Veloce | 🐢 Più lento |
| Richiede scaling | Sì | No | No |

**Nota speaker:** "Teniamo questa tabella in mente quando vedremo i risultati: ogni modello ha il suo spazio."

---

### Slide 20 — class_weight="balanced"

**Titolo:** Gestione dell'imbalance: la nostra scelta

**Recap (compatto):**

Il dataset è sbilanciato (~80% non-churn, ~20% churn). Senza correzione, i modelli tendono a predire sempre "non churn".

**Formula peso:**

$$w_{\text{churn}} = \frac{N}{2 \cdot N_{\text{churn}}} = \frac{6000}{2 \cdot 1223} \approx 2.45$$

**Decisione (evidenziata):**
> Tutti i modelli oggi usano `class_weight='balanced'`: ogni errore su un churner pesa **2.45×** rispetto a un errore su un non-churner.

**Nota speaker:** "Ricordate dalla Lezione 2? Il class_weight corregge lo sbilanciamento senza modificare i dati — a differenza di SMOTE, che su LR non aggiunge valore (confermato empiricamente nel notebook)."

---

### Slide 21 — [DIVIDER 03]

**Sfondo scuro**

> **03**
>
> METRICHE DI VALUTAZIONE

---

### Slide 22 — "Un modello senza metriche è un'opinione"

**Slide citazione (testo grande, centrato):**

> "Un modello senza metriche è un'opinione.
> E un modello con la metrica sbagliata è un'illusione."

**Sottotesto:**
> DummyClassifier: Accuracy = **79.6%**, Recall churn = **0%**
> Il paradosso dell'accuracy su classi sbilanciate.

**Nota speaker:** "Senza numeri, non sappiamo se abbiamo migliorato qualcosa. E con i numeri sbagliati, crediamo di aver migliorato quando non è così."

---

### Slide 23 — Confusion Matrix — anatomia

**Titolo:** Confusion Matrix nel contesto churn

**Matrice 2×2:**

|  | Predetto: Non-churn | Predetto: Churn |
|--|---------------------|-----------------|
| **Reale: Non-churn** | **TN** — Corretto non-intervento | **FP** — Campagna inutile (costo basso: €50/cliente) |
| **Reale: Churn** | **FN** — Churner perso (costo alto: €500–1000/cliente) | **TP** — Churner identificato e salvato! |

**Insight (in box colorato):**
> Il rapporto costo FN/FP è **10–20×**. Perdere un churner costa 10–20 volte di più che inviare una campagna inutile.

**Nota speaker:** "Quale cella è più costosa per la banca? Il falso negativo: un cliente che se ne va senza che potessimo intervenire."

---

### Slide 24 — Metriche derivate dalla matrice

**Titolo:** Metriche di valutazione — reference compatto

| Metrica | Formula | Interpretazione nel churn |
|---------|---------|--------------------------|
| **Accuracy** | $\frac{TP+TN}{N}$ | % predizioni corrette (fuorviante se sbilanciato!) |
| **Precision** | $\frac{TP}{TP+FP}$ | Quanti dei "predetti churn" lo sono davvero |
| **Recall** | $\frac{TP}{TP+FN}$ | Quanti churner reali vengono identificati |
| **F1** | $\frac{2 \cdot P \cdot R}{P + R}$ | Media armonica: penalizza sbilanciamento P/R |
| **ROC-AUC** | Area sotto curva ROC | Capacità discriminativa *indipendente dalla soglia* |

**Nota speaker:** "Queste le conosciamo dalla Lezione 2. Oggi le calcoleremo davvero su tre modelli diversi."

---

### Slide 25 — ROC Curve e AUC — riepilogo

**Titolo:** ROC Curve — come leggerla

**Diagramma ROC con annotazioni:**
- Asse X: False Positive Rate (1 − Specificità)
- Asse Y: True Positive Rate (Recall)
- Diagonale tratteggiata: modello casuale (AUC = 0.5)
- Curva sopra la diagonale: modello utile
- Punto (0,1): modello perfetto (AUC = 1.0)

**Annotazione sulla curva:**
> "Più la curva è in alto a sinistra, migliore è il modello.
> L'AUC riassume tutto in un solo numero: probabilità che il modello
> assegni un punteggio più alto a un positivo che a un negativo scelto a caso."

---

### Slide 26 — [DIVIDER 04]

**Sfondo scuro**

> **04**
>
> CONFRONTO MODELLI

---

### Slide 27 — Risultati: tabella comparativa

**Titolo:** Risultati su Validation Set

| Modello | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|-----|---------|
| Dummy (sempre "non churn") | 0.796 | 0.000 | 0.000 | 0.000 | 0.500 |
| LR (class_weight) | 0.779 | 0.475 | **0.784** | 0.591 | 0.855 |
| DT (depth=6, cw) | 0.775 | 0.469 | 0.777 | 0.585 | 0.843 |
| RF (soglia=0.5) | 0.859 | 0.778 | 0.429 | 0.553 | **0.872** |
| **RF (soglia=0.36)** | **0.865** | **0.685** | **0.623** | **0.652** | **0.872** |

**Insight (box evidenziato):**
> Nessun modello domina su tutte le metriche.
> RF ha la migliore AUC (0.872), ma la soglia 0.5 sacrifica il recall.
> Con soglia ottimale (0.36): F1 = 0.652, il miglior compromesso.

**Nota speaker:** "Ecco i numeri reali. Notiamo che LR ha recall 78% — altissimo — ma precision bassa (47.5%). RF ha la migliore AUC (0.872) e con soglia 0.5 una precision del 77.8%, ma perde recall (42.9%). La soglia fa la differenza."

---

### Slide 28 — Confusion Matrix a confronto

**Titolo:** Quanti churner perdiamo con ciascun modello?

*(3 heatmap side-by-side — inserire `outputs/figures/lesson_03_confusion_matrices.png`)*

**Tabella sotto le heatmap:**

| Modello | TP (identificati) | FN (persi) | su 408 churner |
|---------|-------------------|------------|----------------|
| LR (class_weight) | 320 | 88 | Recall 78.4% |
| DT (depth=6) | 317 | 91 | Recall 77.7% |
| RF (soglia=0.36) | 254 | 154 | Recall 62.3% |

**Nota speaker:** "Guardate gli FN: LR perde 88 churner, RF ne perde 154 ma con meno falsi allarmi. La scelta dipende dal budget di retention della banca."

---

### Slide 29 — ROC Curve: tre modelli sovrapposti

**Titolo:** ROC Curves — Confronto Modelli

*(Inserire `outputs/figures/lesson_03_roc_curve_comparison.png`)*

**Annotazione:**
- RF (AUC = 0.872) — curva blu
- LR (AUC = 0.855) — curva arancio
- DT depth=6 (AUC = 0.843) — curva verde
- Casuale (AUC = 0.500) — diagonale tratteggiata

**Insight:** "RF domina su quasi tutta la curva: a qualsiasi livello di falsi positivi, identifica più churner degli altri."

---

### Slide 30 — "Il modello migliore dipende dalla domanda"

**Titolo:** Nessun vincitore universale

**Due scenari business:**

| Scenario | Obiettivo | Modello consigliato | Soglia | Risultato atteso |
|----------|-----------|-------|--------|---------|
| **"Budget illimitato"** — Contattare tutti i potenziali churner | Massimizzare Recall | LR class_weight | 0.5 | Recall 78%, 320/408 salvati |
| **"Budget limitato"** — Solo interventi mirati con alta probabilità | Massimizzare Precision | RF | soglia alta (0.5+) | Precision 70%+, pochi falsi allarmi |
| **"Compromesso"** — Bilanciare copertura e budget | Massimizzare F1 | RF | 0.36 | F1 0.652, miglior compromesso |

**Citazione:**
> "Non esiste il modello migliore in assoluto. Esiste il modello migliore per il vostro obiettivo di business."

---

### Slide 31 — [DIVIDER 05]

**Sfondo scuro**

> **05**
>
> THRESHOLD TUNING

---

### Slide 32 — Soglia di default vs soglia ottimale

**Titolo:** La soglia di classificazione come leva operativa

**Sinistra (default):**
> Soglia = 0.5 (default sklearn)
> - RF predice "Churn" solo se $\hat{p} > 0.5$
> - Recall = **42.9%** — perde 233 churner su 408
> - Precision = **77.8%** — pochi falsi allarmi

**Destra (ottimizzata):**
> Soglia = 0.36 (ottimizzata su val set per F1)
> - RF predice "Churn" se $\hat{p} > 0.36$
> - Recall = **62.3%** — perde 154 churner
> - Precision = **68.5%** — leggero calo

**Metafora slider:**
```
Precision alta ←━━━━━━━━━●━━━━━━━━━→ Recall alto
             soglia=0.7      soglia=0.36      soglia=0.2
```

**Nota chiave:** La ROC-AUC (0.872) non cambia: è intrinsecamente indipendente dalla soglia. Muovere la soglia = muoversi *lungo* la stessa curva ROC.

**Nota speaker:** "In Lezione 2 avevamo promesso: 'la soglia si sceglie sul Validation Set'. Oggi manteniamo la promessa."

---

### Slide 33 — Curva Precision-Recall

**Titolo:** Precision-Recall trade-off — RF su Validation Set

*(Inserire `outputs/figures/lesson_03_pr_curve.png`)*

**Annotazione sul grafico:**
- Punto rosso: soglia ottimale F1 = 0.36
- Linea grigia tratteggiata: baseline casuale (precision = 0.20)
- Punto soglia 0.5 annotato

**Testo:**
> Abbassando la soglia: più churner identificati (recall ↑), ma più falsi allarmi (precision ↓).
> La soglia F1-ottimale (0.36) bilancia i due obiettivi.

**Domanda per la classe:**
> "Se la banca vuole recall ≥ 70%, quale precision possiamo aspettarci?"

---

### Slide 34 — Risultato: modello finale

**Titolo:** Il nostro modello candidato

**Stat callout grande (centrato):**

> **Random Forest + soglia 0.36**
>
> | | Val | Test |
> |--|-----|------|
> | ROC-AUC | 0.872 | **0.858** |
> | Recall | 0.623 | 0.592 |
> | Precision | 0.685 | 0.620 |
> | F1 | 0.652 | 0.606 |

**Insight:**
> Test AUC = 0.858 (val era 0.872): calo di 0.014 — fisiologico e accettabile.
> Il modello generalizza. Rispetto alla baseline L1 (AUC=0.85, recall=37%):
> **+0.008 AUC e +22 punti percentuali di recall con pipeline corretta.**

**Bridge:**
> "Abbiamo il modello migliore. Ma *perché* predice churn per certi clienti?
> Lo scopriremo nella **Lezione 4 — Interpretabilità** con SHAP e Partial Dependence Plots."

---

### Slide 35 — [DIVIDER 06]

**Sfondo scuro**

> **06**
>
> LIVE CODING

**Sottotitolo (opzionale, come miglioramento rispetto a L1/L2):**
> Cosa coderemo:
> - Caricamento split preprocessati
> - DummyClassifier → il paradosso dell'accuracy
> - LR senza/con class_weight → impatto sullo sbilanciamento
> - DT: curva depth vs metriche (bias-variance)
> - RF + threshold tuning + confronto finale

---

### Slide 36 — Grazie dell'attenzione

**Testo:**

> Grazie dell'attenzione.

| | |
|--|--|
| **Enrico Huber** | **Pietro Soglia** |
| enrico.huber@... | pietro.soglia@... |

*(Stesso layout L1/L2)*

---

## 9) Speaker notes riassuntive per transizioni

| Transizione | Frase suggerita |
|-------------|-----------------|
| Slide 6 → 7 | "Partiamo con un breve riepilogo di dove eravamo rimasti." |
| Slide 10 → 11 | "Bene, sappiamo da dove partiamo. Vediamo gli strumenti che useremo oggi." |
| Slide 20 → 21 | "Abbiamo descritto i tre modelli. Ora stabiliamo come misureremo il loro successo." |
| Slide 25 → 26 | "Le metriche le conosciamo. Vediamo i risultati reali." |
| Slide 30 → 31 | "La scelta del modello non basta: la soglia è un parametro altrettanto importante." |
| Slide 34 → 35 | "Abbiamo il nostro candidato. Ora apriamo il notebook e ripercorriamo tutto il processo con il codice." |

---

## 10) Checklist pre-produzione slide

- [ ] Eseguire `notebooks/lesson-03/lesson_03.py` per generare tutti i plot in `outputs/figures/`
- [ ] Verificare che `lesson_03_metrics.json` sia aggiornato (valori usati nelle slide)
- [ ] Inserire le figure nei placeholder: slide 28, 29, 33
- [ ] Aggiornare foto/contatti docenti se necessario
- [ ] Verificare che tutti i numeri nelle slide corrispondano all'output del notebook
- [ ] Controllare formattazione formule LaTeX nel tool di presentazione
- [ ] Rispettare il limite di 800 caratteri per slide (tabelle incluse)

---

## File ispezionati

- `lessons/lesson-01-churn-classification-eda/lesson_01.pdf` (36 pagine, testo completo estratto)
- `lessons/lesson-02-preprocessing/lesson_02.pdf` (36 pagine, testo completo estratto)
- `notebooks/lesson-01/lesson_01.ipynb` (outline, sezione baseline, riepilogo)
- `notebooks/lesson-02/lesson_02.ipynb` (contenuto completo — tutte le 17 sezioni)
- `notebooks/lesson-02/lesson_02_live_coding.ipynb` (sommario struttura)
- `notebooks/lesson-03/lesson_03.py` (contenuto completo — tutti i blocchi A-G)
- `outputs/data/lesson_02_feature_names.json` (30 nomi feature)
- `outputs/data/lesson_03_metrics.json` (metriche finali reali)
- `outputs/data/` (tutti i file Parquet + preprocessor.pkl)
- `outputs/figures/` (tutti i PNG plot esistenti da L1, L2 e L3)
- `outputs/models/` (vuoto)
- `outputs/config/lesson_01_eda_summary.json`
- `README.md`
- `.github/copilot-instructions.md`
- `.github/skills/lesson-discovery/SKILL.md`
- `.github/instructions/python.instructions.md`
