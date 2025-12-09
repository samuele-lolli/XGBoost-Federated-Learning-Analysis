# Federated Learning for XGBoost: Non-IID Analysis

Questo repository contiene il codice specifico per l'analisi sperimentale della tesi focalizzata sull'applicazione del Federated Learning a modelli ad albero (XGBoost) su dati tabulari.

## Obiettivi

Il progetto confronta due tecniche di aggregazione specifiche per XGBoost:

| Aspetto | FedBagging | FedCyclic |
| :--- | :--- | :--- |
| **Metodo** | Ensemble di alberi addestrati in parallelo | Raffinamento sequenziale del modello |
| **Punti di Forza (Risultato)** | Maggiore stabilità in scenari Non-IID  | Convergenza rapida e performance superiore in scenari IID |

## ⚙️ Installazione e Utilizzo

**Tutta la configurazione, l'installazione delle dipendenze (Flower, PyTorch, XGBoost) e la logica di esecuzione sono definite nel repository principale.**

Per il progetto principale, consultare:

➡️ **[Federated Learning: Aggregation Strategies, Heterogeneity & Robustness](https://github.com/samuele-lolli/Aggregation-Federated-Learning-Analysis/)**
