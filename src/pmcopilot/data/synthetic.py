"""
synthetic.py — generatore di dati "in stile AI4I 2020".

PERCHÉ ESISTE
    Il dataset reale (AI4I 2020, UCI Machine Learning Repository) è a sua volta
    SINTETICO: gli autori lo hanno generato con regole fisiche documentate nel
    paper di Matzka (2020). Qui riproduciamo quelle regole, in modo approssimato,
    per due motivi:
      1. il progetto deve funzionare anche senza connessione (o nei test/CI);
      2. leggere il generatore è il modo migliore per capire PERCHÉ un guasto
         avviene, e quindi perché le feature di domain.py funzionano.

    Se il download riesce (vedi download.py) si usano i dati veri; altrimenti
    questi. Le colonne sono identiche, quindi il resto del codice non cambia.

COSA IMPARI QUI
    * numpy: generatori casuali riproducibili (default_rng), random walk con
      cumsum, operazioni vettoriali con maschere booleane (&, |), np.clip.
    * pandas: costruire un DataFrame da array numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmcopilot import domain

# Nomi delle colonne ESATTAMENTE come nel CSV originale di UCI.
AI4I_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def _standardized_random_walk(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    """Random walk normalizzato a media 0 e deviazione standard `sd`.

    Una random walk è la somma cumulativa di passi casuali: ogni valore è il
    precedente più un piccolo rumore. È un buon modello per una temperatura
    ambiente, che varia lentamente invece di saltare da un valore all'altro.
    """
    steps = rng.normal(loc=0.0, scale=1.0, size=n)
    walk = np.cumsum(steps)  # [s0, s0+s1, s0+s1+s2, ...]
    walk = (walk - walk.mean()) / walk.std()  # standardizzazione (z-score)
    return walk * sd


def generate_ai4i_like(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Genera `n` lavorazioni con le stesse colonne del dataset AI4I 2020."""
    rng = np.random.default_rng(seed)  # stesso seed -> stessi dati, sempre

    # --- 1. tipo di prodotto: 50% L, 30% M, 20% H (come nel dataset reale) ---
    types = rng.choice(np.array(domain.MACHINE_TYPES), size=n, p=[0.5, 0.3, 0.2])
    serials = rng.integers(10_000, 60_000, size=n)
    product_ids = np.char.add(types.astype(str), serials.astype(str))

    # --- 2. temperature ---------------------------------------------------------
    # Aria: ~300 K con variazioni lente (sd 2 K).
    air_temp = 300.0 + _standardized_random_walk(rng, n, sd=2.0)
    # Processo: aria + 10 K + un'altra variazione lenta (sd 1 K).
    process_temp = air_temp + 10.0 + _standardized_random_walk(rng, n, sd=1.0)
    # Arrotondiamo SUBITO a un decimale, come nel CSV reale: così le regole di
    # guasto sotto vengono calcolate sugli stessi valori che il modello vedrà.
    air_temp = air_temp.round(1)
    process_temp = process_temp.round(1)

    # --- 3. coppia e velocità ----------------------------------------------------
    # Coppia ~ Normale(40 Nm, 10 Nm), senza valori troppo bassi.
    torque = np.clip(rng.normal(40.0, 10.0, size=n), 3.8, None).round(1)
    # Nel dataset reale velocità e coppia sono fortemente anticorrelate
    # (a potenza circa costante, più coppia = meno giri). Lo imitiamo così:
    rpm = 1540.0 - 16.0 * (torque - 40.0) + rng.normal(0.0, 80.0, size=n)
    rpm = np.clip(rpm, 1168, 2886).round().astype(int)

    # --- 4. usura utensile e guasto TWF -----------------------------------------
    # Questo è l'unico pezzo con un ciclo for: l'usura dipende dalla riga
    # precedente (stato), quindi non si vettorizza in modo semplice.
    tool_wear = np.zeros(n, dtype=int)
    twf = np.zeros(n, dtype=int)
    current_wear = 0
    replace_at = rng.integers(domain.TOOL_WEAR_REPLACEMENT_MIN, domain.TOOL_WEAR_REPLACEMENT_MAX + 1)
    for i, product_type in enumerate(types):
        tool_wear[i] = current_wear
        current_wear += domain.TOOL_WEAR_INCREMENT[str(product_type)]
        if tool_wear[i] >= replace_at:
            # Nel dataset originale, a fine vita l'utensile si rompe ~42% delle
            # volte (51 su 120), altrimenti viene sostituito prima del guasto.
            if rng.random() < 51 / 120:
                twf[i] = 1
            current_wear = 0
            replace_at = rng.integers(
                domain.TOOL_WEAR_REPLACEMENT_MIN, domain.TOOL_WEAR_REPLACEMENT_MAX + 1
            )

    # --- 5. le altre modalità di guasto: regole vettoriali con maschere ----------
    temp_diff = process_temp - air_temp
    power_w = torque * rpm * 2.0 * np.pi / 60.0  # P = C * omega

    hdf = (temp_diff < domain.HDF_MAX_TEMP_DIFF_K) & (rpm < domain.HDF_MAX_RPM)
    pwf = (power_w < domain.PWF_MIN_POWER_W) | (power_w > domain.PWF_MAX_POWER_W)
    # Soglia OSF diversa per tipo: np.vectorize applica il dizionario elemento per elemento.
    osf_limit = np.vectorize(domain.OSF_LIMIT.get)(types)
    osf = tool_wear * torque > osf_limit
    rnf = rng.random(n) < domain.RNF_PROBABILITY

    machine_failure = (twf.astype(bool) | hdf | pwf | osf | rnf).astype(int)

    # --- 6. assemblaggio del DataFrame ------------------------------------------
    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Machine failure": machine_failure,
            "TWF": twf,
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )
    return df[AI4I_COLUMNS]
