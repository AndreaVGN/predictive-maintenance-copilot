# 1. Dati con pandas e numpy

**File:** `domain.py`, `data/synthetic.py`, `data/download.py`, `data/features.py`, `analysis/eda.py`, `notebooks/01_eda.ipynb`

## Il dataset AI4I 2020

AI4I 2020 (UCI Machine Learning Repository) contiene 10.000 lavorazioni di una fresatrice. Per ognuna ci sono cinque misure: temperatura dell'aria, temperatura di processo, giri, coppia e usura utensile. In più c'è il tipo di prodotto (L, M o H) e l'indicazione di guasto, suddivisa in cinque modalità.

È un dataset **sintetico**, generato con regole fisiche dichiarate dagli autori. Per un progetto didattico è un vantaggio, perché le cause dei guasti sono note e possiamo verificare se il modello le "scopre".

| Modalità | Regola |
|---|---|
| TWF usura utensile | fine vita tra 200 e 240 min |
| HDF dissipazione termica | delta termico < 8,6 K **e** giri < 1380 |
| PWF potenza | potenza < 3500 W **o** > 9000 W |
| OSF sovraccarico | usura × coppia > 11000 / 12000 / 13000 (L/M/H) |
| RNF casuale | 0,1% |

Queste soglie stanno in **`domain.py`**, l'unica fonte di verità usata da generatore, predittore, manuali e test.

## Flusso del codice

```mermaid
flowchart LR
    A[download.py<br>UCI o sintetico] --> B[features.py<br>clean_columns]
    B --> C[database.py<br>build_database]
    C --> D[(plant.db)]
    D --> E[features.py<br>add_engineered_features]
    E --> F[eda.py / train.py]
```

## numpy: cosa guardare nel codice

**Random generator riproducibile** (`synthetic.py`)
```python
rng = np.random.default_rng(seed)       # stesso seed -> stessi numeri, sempre
torque = np.clip(rng.normal(40.0, 10.0, size=n), 3.8, None)
```

**Random walk con `cumsum`**: la temperatura ambiente varia lentamente. Ogni valore è il precedente più un piccolo passo:
```python
walk = np.cumsum(rng.normal(0, 1, n))
```

**Maschere booleane e operatori vettoriali.** Un'intera regola di guasto in una riga, su 10.000 elementi, senza cicli:
```python
hdf = (temp_diff < 8.6) & (rpm < 1380)          # & = AND elemento per elemento
pwf = (power_w < 3500) | (power_w > 9000)       # | = OR
```

**Feature fisiche** (`features.py`):
```python
out["power_w"] = torque * rpm * 2 * np.pi / 60   # P = C · ω
```

!!! question "Perché il ciclo for per l'usura?"
    L'usura di una riga dipende da quella precedente (stato). Le operazioni vettoriali lavorano su elementi indipendenti, quindi qui un ciclo esplicito è la scelta più chiara. Vettorizzare non è sempre possibile né necessario.

## pandas: cosa guardare nel codice

| Operazione | Dove | Equivalente SQL |
|---|---|---|
| `df.rename(columns=...)` | `clean_columns` | `SELECT col AS nuovo` |
| `df.groupby("machine_type").agg(...)` | `eda.failure_rate_by_type` | `GROUP BY` |
| `df[df["machine_failure"] == 1]` | `eda.plot_distributions` | `WHERE` |
| `pd.read_sql_query(sql, conn)` | `database.load_readings` | — (ponte SQL → pandas) |
| `df.to_sql(...)` | `database.build_database` | `INSERT` |

## Data leakage

Le colonne `twf`, `hdf`, `pwf`, `osf`, `rnf` dicono già se c'è un guasto. Se finissero tra le feature, il modello avrebbe un'accuratezza quasi perfetta e sarebbe inutile in produzione, dove quelle colonne non esistono *prima* del guasto. `features.feature_columns()` le esclude e un test lo verifica (`test_feature_columns_never_include_leakage`).

## Esercizi

1. **numpy**: scrivi `temp_diff_zscore(df)` che restituisce lo z-score del delta termico per tipo di macchina, usando `groupby(...).transform`.
2. **pandas**: calcola, per ogni macchina, la lavorazione con la potenza massima (`idxmax`).
3. **Generatore**: cambia in `domain.py` la soglia HDF a 9,0 K, rigenera i dati sintetici (`PMC_FORCE_SYNTHETIC_DATA=true pmc data --force`) e osserva come cambia il numero di guasti HDF.
4. **Riflessione**: il dataset reale ha 339 guasti su 10.000, il generatore circa 520. Quale regola produce la differenza? (Suggerimento: confronta `eda.failure_mode_counts` nei due casi.)
