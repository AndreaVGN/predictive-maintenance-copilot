"""
bm25.py — ricerca per parole chiave con Okapi BM25, scritta da zero (~50 righe).

PERCHÉ SERVE ACCANTO AGLI EMBEDDING
    Gli embedding catturano il SIGNIFICATO ("surriscaldamento" ~ "temperatura
    troppo alta") ma sono deboli sulle corrispondenze ESATTE: un codice di
    allarme "AL-207", una soglia "8,6", una sigla "P1". BM25 fa l'opposto:
    conta le parole in comune. Usarli insieme (ricerca ibrida) prende il
    meglio dei due.

LA FORMULA, PEZZO PER PEZZO
    Per una query q e un documento d:

        score(d, q) = Σ_t  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

    f(t,d)   quante volte il termine t compare in d (term frequency)
    IDF(t)   quanto è raro t nella collezione: ln(1 + (N - n_t + 0.5)/(n_t + 0.5))
             "il" compare ovunque -> IDF ~ 0; "AL-207" compare una volta -> IDF alto
    k1       saturazione: la 10a occorrenza di una parola conta meno della 1a (default 1.5)
    b        normalizzazione per lunghezza: i documenti lunghi non vincono solo
             perché contengono più parole (default 0.75)

    È la stessa famiglia di BM25F (la variante multi-campo), usata nei motori
    di ricerca prima delle reti neurali e ancora oggi una baseline fortissima.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

# Poche parole italiane frequentissime e prive di contenuto ("stopword").
STOPWORDS = frozenset(
    "il lo la i gli le un uno una di a da in con su per tra fra e o ed che non si "
    "del dello della dei degli delle al allo alla ai agli alle dal dallo dalla dai "
    "dagli dalle nel nello nella nei negli nelle sul sullo sulla sui sugli sulle "
    "è sono come cosa quale quali quando quanto quanti dove deve devo fare fa "
    "ci ne se mi ti più anche questo questa quello quella".split()
)

# Un token è una sequenza alfanumerica, che può contenere "-", "," o "." interni:
# così "AL-207", "8,6" e "3.500" restano interi.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-,.][a-z0-9]+)*")


def light_stem(token: str) -> str:
    """Stemming "leggero" per l'italiano: toglie la vocale finale delle parole lunghe.

    "intervento", "interventi", "interventa" -> "intervent". È rozzo (uno stemmer
    vero come Snowball gestisce molti più suffissi) ma recupera i casi più comuni
    di singolare/plurale e maschile/femminile senza dipendenze.
    """
    if len(token) > 4 and token.isalpha() and token[-1] in "aeio":
        return token[:-1]
    return token


def _keep(tok: str) -> bool:
    return tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())


def tokenize(text: str) -> list[str]:
    """Minuscole, senza accenti, senza stopword, con stemming leggero.

    I token composti ("al-207", "8,6") restano interi E vengono aggiunte le parti,
    così "AL-207" trova sia "AL-207" sia "allarme 207".
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        if not _keep(tok):
            continue
        tokens.append(light_stem(tok))
        if any(sep in tok for sep in "-,."):
            tokens.extend(light_stem(p) for p in re.split(r"[-,.]", tok) if p and _keep(p))
    return tokens


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for counts in self.docs:
            df.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - nt + 0.5) / (nt + 0.5)) for t, nt in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            out.append(sum(
                self.idf.get(t, 0.0) * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                for t in terms if counts[t]
            ))
        return out


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fonde più classifiche: score(d) = Σ 1 / (k + posizione di d in ciascuna).

    Non serve normalizzare punteggi su scale diverse (distanza coseno vs BM25):
    conta solo la POSIZIONE. k=60 smorza il peso delle prime posizioni ed è il
    valore del paper originale (Cormack et al., 2009).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
