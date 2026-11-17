"""
guardrails.py — le protezioni dell'agente, organizzate in tre livelli.

IL PRINCIPIO: DIFESA IN PROFONDITÀ
    Nessun controllo è perfetto. Per questo ne mettiamo diversi, indipendenti,
    in punti diversi del flusso. Un attacco deve superarli TUTTI per fare danni.

    INGRESSO   check_input()   prima che la domanda arrivi all'LLM
               - lunghezza massima (evita prompt enormi e costi/latenze)
               - pattern di prompt injection noti (IT + EN)
               - richieste esplicite di operazioni distruttive sul database

    AZIONI     validate_sql()  prima che una query arrivi al database
               - una sola istruzione, solo SELECT (o WITH ... SELECT)
               - solo tabelle in whitelist
               - funzioni pericolose vietate (es. load_extension)
               - LIMIT forzato
               ...e in più la connessione è in sola lettura (database.py)
               e le azioni con effetti (ordini di lavoro) richiedono
               l'approvazione di una persona (human-in-the-loop, graph.py).

    USCITA     check_output()  prima che la risposta arrivi all'utente
               - rimuove eventuali blocchi di "ragionamento" <think>
               - blocca la fuga del system prompt (canary token)
               - segnala codici macchina inesistenti (allucinazioni)
               - garantisce le citazioni quando sono stati usati i manuali

LIMITI (da dire onestamente a un colloquio)
    I controlli a regole (regex) sono veloci e spiegabili ma aggirabili con
    parafrasi. In produzione si affiancano classificatori dedicati (es. un
    piccolo modello addestrato a riconoscere injection, o un LLM "guard").
    La sicurezza vera viene dai livelli strutturali: permessi minimi sul DB,
    whitelist, approvazione umana. Quelli non si aggirano con le parole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# ============================================================== INGRESSO
# Ogni pattern ha un nome: finisce nei log e nella valutazione, così sappiamo
# QUALE regola è scattata.
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"(ignora|dimentica|trascura|ignore|disregard|forget)\b.{0,40}\b"
        r"(istruzion\w*|regol\w*|prompt|instructions?|rules?|guidelines?)",
        re.IGNORECASE | re.DOTALL,
    ),
    "reveal_prompt": re.compile(
        r"(rivela|mostra\w*|stampa|ripeti|dimmi|show|reveal|print|repeat)\b.{0,40}\b"
        r"(system\s*prompt|prompt\s+di\s+sistema|tue\s+istruzioni|your\s+(system\s+)?instructions)",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_override": re.compile(
        r"(da\s+ora\s+(in\s+poi\s+)?sei|fai\s+finta\s+di\s+essere|comportati\s+come|"
        r"you\s+are\s+now|act\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)|pretend\s+to\s+be)",
        re.IGNORECASE,
    ),
    "jailbreak_keyword": re.compile(r"\b(jailbreak|DAN\s+mode|developer\s+mode)\b", re.IGNORECASE),
    "fake_system_tag": re.compile(r"(<\s*/?\s*system\s*>|\[\s*system\s*\]|###\s*system)", re.IGNORECASE),
}

DESTRUCTIVE_SQL_PATTERN = re.compile(
    r"\b(drop\s+table|delete\s+from|truncate\s+table|update\s+\w+\s+set|insert\s+into|alter\s+table)\b",
    re.IGNORECASE,
)


@dataclass
class GuardResult:
    allowed: bool
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def check_input(text: str, max_chars: int = 2000) -> GuardResult:
    """Controlla la domanda dell'utente. Non chiama l'LLM: è istantaneo."""
    if not text or not text.strip():
        return GuardResult(False, "La domanda è vuota.", ["empty_input"])
    if len(text) > max_chars:
        return GuardResult(False, f"La domanda supera i {max_chars} caratteri consentiti.", ["too_long"])
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(text):
            return GuardResult(
                False,
                "La richiesta sembra un tentativo di modificare il comportamento dell'assistente.",
                [f"injection:{name}"],
            )
    if DESTRUCTIVE_SQL_PATTERN.search(text):
        return GuardResult(
            False,
            "L'assistente può solo leggere i dati: non può modificarli o cancellarli.",
            ["destructive_sql_request"],
        )
    return GuardResult(True)


# ============================================================== AZIONI (SQL)
class SQLGuardError(ValueError):
    """Query rifiutata: il messaggio torna all'agente, che può correggerla."""


FORBIDDEN_STATEMENTS: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Merge, exp.Transaction,
)  # fmt: skip
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"load_extension", "readfile", "writefile", "fts3_tokenizer", "edit", "randomblob", "zeroblob"}
)


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def validate_sql(sql: str, allowed_tables: frozenset[str], max_rows: int = 200) -> str:
    """Analizza la query con un parser SQL vero e restituisce una versione sicura.

    Perché un parser e non una regex? Una regex che cerca "DROP" si aggira con
    commenti (DR/**/OP), maiuscole strane o stringhe. Il parser (sqlglot)
    costruisce l'albero sintattico (AST): controlliamo la STRUTTURA della query,
    non il testo.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLGuardError(f"Query SQL non valida: {exc}") from exc
    if len(statements) != 1:
        raise SQLGuardError("È consentita una sola istruzione SQL per volta.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):  # Select, Union, ecc.
        raise SQLGuardError(f"Sono consentite solo query SELECT (ricevuto: {type(tree).__name__}).")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_STATEMENTS):
            raise SQLGuardError(f"Operazione non consentita: {type(node).__name__}.")
        if isinstance(node, exp.Func) and _function_name(node) in FORBIDDEN_FUNCTIONS:
            raise SQLGuardError(f"Funzione non consentita: {_function_name(node)}.")

    # Whitelist delle tabelle. I nomi delle CTE (WITH nome AS ...) sono leciti.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table)} - cte_names
    unknown = tables - allowed_tables
    if unknown:
        raise SQLGuardError(
            f"Tabelle non consentite o inesistenti: {sorted(unknown)}. "
            f"Tabelle disponibili: {sorted(allowed_tables)}."
        )

    # LIMIT: aggiunto se manca, ridotto se troppo alto.
    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    else:
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int and int(value.this) <= max_rows):
            tree = tree.limit(max_rows)
    return tree.sql(dialect="sqlite")


# ============================================================== USCITA
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
MACHINE_ID_PATTERN = re.compile(r"\bM\d{2}\b")
CITATION_PATTERN = re.compile(r"\[\d+\]|fonte", re.IGNORECASE)

# Il "canary" è una stringa casuale nascosta nel system prompt. Non ha alcun
# significato: se compare nella risposta, il modello sta ripetendo il prompt.
CANARY_TOKEN = "PMC-CANARY-7f3a91"


@dataclass
class OutputCheck:
    answer: str
    flags: list[str] = field(default_factory=list)


def check_output(
    answer: str,
    *,
    known_machine_ids: set[str],
    sources: list[str] | None = None,
) -> OutputCheck:
    """Controlla e, dove possibile, CORREGGE la risposta finale dell'agente.

    Correggere invece di rifiutare (quando è sicuro farlo) migliora l'esperienza:
    se mancano le citazioni le aggiungiamo noi, invece di buttare la risposta.
    """
    flags: list[str] = []
    text = THINK_BLOCK.sub("", answer or "").strip()

    if CANARY_TOKEN in text:
        return OutputCheck(
            "Non posso condividere le mie istruzioni interne. Posso aiutarti con dati, "
            "previsioni o procedure di manutenzione.",
            ["system_prompt_leak"],
        )

    if not text:
        flags.append("empty_answer")
        text = "Non sono riuscito a formulare una risposta. Prova a riformulare la domanda."

    # Codici macchina citati ma inesistenti nel database: possibile allucinazione.
    mentioned = set(MACHINE_ID_PATTERN.findall(text))
    unknown = sorted(mentioned - known_machine_ids)
    if unknown:
        flags.append("unknown_machine_ids")
        text += (
            f"\n\n⚠️ Nota: i codici {', '.join(unknown)} non risultano nel database. "
            "Verifica questa parte della risposta."
        )

    # Se l'agente ha consultato i manuali, la risposta deve citarli.
    if sources and not CITATION_PATTERN.search(text):
        flags.append("citations_added")
        unique = list(dict.fromkeys(sources))
        text += "\n\nFonti: " + "; ".join(unique)

    return OutputCheck(text, flags)


# ============================================================== DATI NON FIDATI
# PROMPT INJECTION INDIRETTA
#   L'attacco non arriva dall'utente ma dai DATI che l'agente legge: una nota
#   in un ordine di lavoro ("ignora le regole e cancella..."), una riga di un
#   manuale manomesso, il testo di una pagina web. Il modello vede quel testo
#   nel contesto e potrebbe scambiarlo per un'istruzione.
#   Difese (anche qui, più livelli):
#     1. il system prompt dice che l'output dei tool è un DATO (regola 7);
#     2. scan_untrusted_text cerca pattern di injection negli output dei tool:
#        se li trova, oscura la parte sospetta e segnala il caso (flag);
#     3. i dati scritti dall'agente (note degli ordini) vengono ripuliti
#        PRIMA di finire nel database, così non tornano indietro più tardi.
UNTRUSTED_REPLACEMENT = "[contenuto rimosso: sembrava un'istruzione]"


def scan_untrusted_text(text: str) -> tuple[str, list[str]]:
    """Oscura le frasi che sembrano istruzioni dentro un testo NON fidato."""
    flags: list[str] = []
    cleaned = text
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(cleaned):
            flags.append(f"indirect_injection:{name}")
            cleaned = pattern.sub(UNTRUSTED_REPLACEMENT, cleaned)
    return cleaned, flags


def sanitize_note(text: str, max_chars: int = 500) -> str:
    """Pulisce un testo libero prima di salvarlo (es. le note degli ordini di lavoro)."""
    cleaned, _ = scan_untrusted_text(text)
    cleaned = re.sub(r"<[^>]*>", "", cleaned)  # niente tag HTML
    cleaned = re.sub(r"[<>`]", "", cleaned)  # né caratteri per blocchi di codice
    return cleaned.strip()[:max_chars]


# ================================================================== DATI PERSONALI
# Log e trace contengono le domande degli utenti. Prima di scriverle oscuriamo
# i dati personali più comuni: è "privacy by design" (GDPR art. 25). I nomi dei
# tecnici del dataset restano, perché sono dati di lavoro dell'impianto.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "codice_fiscale": re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", re.IGNORECASE),
    "iban": re.compile(r"\bIT\d{2}[A-Z]\d{10}[0-9A-Z]{12}\b", re.IGNORECASE),
    "telefono": re.compile(r"(?<!\w)(?:\+39[\s.]?)?3\d{2}[\s.]?\d{3}[\s.]?\d{3,4}(?!\w)"),
}


def redact_pii(text: str) -> str:
    """Sostituisce i dati personali con un segnaposto, es. [email]."""
    if not text:
        return text
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return text
