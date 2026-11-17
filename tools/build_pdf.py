"""
build_pdf.py — costruisce la guida didattica in PDF a partire dai Markdown.

    python tools/build_pdf.py            (oppure: make pdf)
    -> docs/guida/guida.pdf

COME FUNZIONA
    1. Legge i capitoli docs/guida/NN-*.md in ordine di nome.
    2. Pre-elabora il Markdown:
         {{code: src/pmcopilot/ml/drift.py::psi}}       -> il codice VERO della funzione
         {{code: src/pmcopilot/ml/drift.py::Classe.metodo}}
         {{code: src/pmcopilot/api/static/app.js:40-60}} -> un intervallo di righe
       Gli snippet sono estratti dal codice al momento della build (con il
       modulo ast per Python), quindi la guida non va mai fuori sincrono.
         ```mermaid ... ```  -> diagramma SVG (con mmdc, la CLI di Mermaid;
       gli SVG già generati sono in docs/guida/img/mermaid/ e vengono riusati).
    3. Markdown -> HTML con python-markdown (tabelle, note, riquadri, codice
       colorato da Pygments) e un foglio di stile per la stampa.
    4. HTML -> PDF con Chromium (Playwright), in due passate:
         la prima per sapere a che pagina cade ogni titolo,
         la seconda per scrivere i numeri di pagina nell'indice.
       La copertina è un PDF a parte (senza numero di pagina) unito con pypdf,
       che aggiunge anche i segnalibri (il pannello laterale dei lettori PDF).

DIPENDENZE:  pip install -e ".[pdf]"  e  playwright install chromium
             (mermaid-cli serve solo se aggiungi o modifichi diagrammi:
              npm install -g @mermaid-js/mermaid-cli)
"""

from __future__ import annotations

import ast
import hashlib
import html
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guida"
IMG = GUIDE / "img"
MERMAID_DIR = IMG / "mermaid"
OUTPUT = GUIDE / "guida.pdf"
TITLE = "Predictive Maintenance Copilot"
SUBTITLE = "Guida completa al progetto, dalle basi alla produzione"


# ================================================================ snippet
def _python_snippet(path: Path, qualname: str) -> str:
    """Sorgente di una funzione o classe (anche Classe.metodo) con il modulo ast."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node: ast.AST = tree
    for part in qualname.split("."):
        found = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and child.name == part:
                found = child
                break
            # costanti a livello di modulo: NOME = ...
            if isinstance(child, ast.Assign | ast.AnnAssign):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                if any(isinstance(t, ast.Name) and t.id == part for t in targets):
                    found = child
                    break
        if found is None:
            raise KeyError(f"{qualname} non trovato in {path}")
        node = found
    lines = source.splitlines()
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    segment = lines[start:node.end_lineno]
    indent = min((len(line) - len(line.lstrip()) for line in segment if line.strip()), default=0)
    return "\n".join(line[indent:] for line in segment)


LANG_BY_SUFFIX = {".py": "python", ".js": "javascript", ".css": "css", ".html": "html", ".json": "json",
                  ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".md": "markdown", ".sql": "sql",
                  ".jsonl": "json", "": "makefile"}


def _expand_code(match: re.Match[str]) -> str:
    spec = match.group(1).strip()
    if "::" in spec:
        rel, name = spec.split("::", 1)
        code = _python_snippet(ROOT / rel, name)
    else:
        rel, _, rng = spec.rpartition(":")
        if not rel or not re.fullmatch(r"\d+-\d+", rng):
            rel, rng = spec, ""
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        if rng:
            a, b = (int(x) for x in rng.split("-"))
            lines = lines[a - 1:b]
        code = "\n".join(lines)
    lang = LANG_BY_SUFFIX.get(Path(rel).suffix, "text")
    return f'<p class="code-path">{html.escape(rel)}</p>\n\n```{lang}\n{code}\n```\n'


# ================================================================ mermaid
MERMAID_CONFIG = {
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Inter, Helvetica Neue, Arial, sans-serif", "fontSize": "15px",
        "primaryColor": "#e8f0fa", "primaryBorderColor": "#1d5c9e", "primaryTextColor": "#1e262d",
        "lineColor": "#5b6872", "secondaryColor": "#fff6d6", "tertiaryColor": "#f3f5f6",
        "noteBkgColor": "#fff6d6", "noteBorderColor": "#e0b400",
        "actorBkg": "#e8f0fa", "actorBorder": "#1d5c9e", "signalColor": "#1e262d",
    },
}


def _render_mermaid(match: re.Match[str]) -> str:
    source = match.group(1).strip()
    caption = ""
    first = source.splitlines()[0]
    if first.startswith("%% "):  # prima riga "%% Didascalia" -> didascalia della figura
        caption = first[3:].strip()
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    svg = MERMAID_DIR / f"{digest}.svg"
    if not svg.exists():
        mmdc = shutil.which("mmdc")
        if not mmdc:
            sys.exit(f"Manca il diagramma {svg.name} e mmdc non è installato "
                     "(npm install -g @mermaid-js/mermaid-cli).")
        MERMAID_DIR.mkdir(parents=True, exist_ok=True)
        import json

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "d.mmd"
            cfg = Path(tmp) / "c.json"
            src.write_text(source, encoding="utf-8")
            cfg.write_text(json.dumps(MERMAID_CONFIG))
            subprocess.run([mmdc, "-i", str(src), "-o", str(svg), "-c", str(cfg), "-b", "transparent"],
                           check=True, capture_output=True)
    rel = svg.relative_to(GUIDE).as_posix()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'\n<figure class="diagram"><img src="{rel}" alt="{html.escape(caption)}">{cap}</figure>\n'


# ================================================================ markdown
def preprocess(text: str) -> str:
    text = re.sub(r"```mermaid\n(.*?)```", _render_mermaid, text, flags=re.S)
    return re.sub(r"\{\{code:\s*(.+?)\}\}", _expand_code, text)


def to_html(text: str) -> str:
    md = markdown.Markdown(extensions=[
        "extra", "admonition", "sane_lists", "toc",
        "codehilite",
    ], extension_configs={
        "codehilite": {"guess_lang": False, "css_class": "hl"},
        "toc": {"permalink": False},
    })
    out = md.convert(text)
    # <p><img alt="X" src="..."></p>  ->  figura con didascalia
    out = re.sub(r'<p>\s*<img alt="([^"]*)" src="([^"]+)"\s*/?>\s*</p>',
                 lambda m: (f'<figure><img src="{m.group(2)}" alt="{m.group(1)}">'
                            f'<figcaption>{m.group(1)}</figcaption></figure>'), out)
    return out


def chapters() -> list[Path]:
    return sorted(p for p in GUIDE.glob("[0-9][0-9]-*.md"))


# ================================================================ indice
MARK = "ZQX{:04d}QXZ"  # testo invisibile nei titoli: serve a trovarli nel PDF


def build_body(page_numbers: dict[int, int] | None) -> tuple[str, list[tuple[int, int, str, str]]]:
    """HTML di indice + capitoli. Restituisce anche l'elenco dei titoli (id, livello, testo, ancora)."""
    parts, headings = [], []
    counter = 0
    for n, path in enumerate(chapters()):
        body = to_html(preprocess(path.read_text(encoding="utf-8")))

        def tag(m: re.Match[str], n: int = n) -> str:  # n legato qui: niente sorprese nel ciclo
            nonlocal counter
            level, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
            anchor = re.search(r'id="([^"]+)"', attrs)
            anchor_id = f"c{n}-{anchor.group(1)}" if anchor else f"c{n}-h{counter}"
            text = re.sub(r"<[^>]+>", "", inner)
            headings.append((counter, level, html.unescape(text), anchor_id))
            marker = f'<span class="mark">{MARK.format(counter)}</span>'
            counter += 1
            return f'<h{level} id="{anchor_id}">{inner}{marker}</h{level}>'

        body = re.sub(r'<h([12])([^>]*)>(.*?)</h\1>', tag, body, flags=re.S)
        parts.append(f'<section class="chapter">{body}</section>')

    toc = ['<nav class="toc"><h1 class="toc-title">Indice</h1><ol>']
    for idx, level, text, anchor in headings:
        page = page_numbers.get(idx, 0) if page_numbers else 0
        toc.append(f'<li class="l{level}"><a href="#{anchor}"><span>{html.escape(text)}</span>'
                   f'<span class="dots"></span><span class="pg">{page or "000"}</span></a></li>')
    toc.append("</ol></nav>")
    return "\n".join(toc) + "\n".join(parts), headings


def page_html(body: str) -> str:
    css = (GUIDE / "guida.css").read_text(encoding="utf-8")
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".hl")
    return (f'<!doctype html><html lang="it"><head><meta charset="utf-8"><title>{TITLE}</title>'
            f"<style>{pygments_css}\n{css}</style></head><body>{body}</body></html>")


def cover_html() -> str:
    css = (GUIDE / "guida.css").read_text(encoding="utf-8")
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><style>{css}</style></head>
<body class="cover"><div class="cover-band"></div>
<div class="cover-body">
  <p class="cover-kicker">Progetto di portfolio · AI &amp; Data Engineering</p>
  <h1>{TITLE}</h1>
  <p class="cover-sub">{SUBTITLE}</p>
  <ul class="cover-list">
    <li>numpy, pandas, matplotlib, SQL</li><li>machine learning con scikit-learn e drift</li>
    <li>RAG ibrido, agente LangGraph, guardrail</li><li>FastAPI, sicurezza, osservabilità, valutazione</li>
  </ul>
</div>
<p class="cover-foot">Andrea Vagnoli · edizione del {date.today():%d/%m/%Y}</p>
<img class="cover-img" src="img/screens/copilot.png" alt="">
</body></html>"""


# ================================================================ PDF
def render(page, html_text: str, out: Path, numbered: bool) -> None:
    tmp_html = GUIDE / "_build.html"  # nella cartella della guida: i percorsi img/ funzionano
    tmp_html.write_text(html_text, encoding="utf-8")
    page.goto(tmp_html.as_uri(), wait_until="networkidle")
    footer = ('<div style="width:100%;font:8px Helvetica,Arial,sans-serif;color:#5b6872;'
              'padding:0 16mm;display:flex;justify-content:space-between">'
              f'<span>{TITLE}</span><span class="pageNumber"></span></div>')
    page.pdf(path=str(out), format="A4", print_background=True, prefer_css_page_size=True,
             display_header_footer=numbered, header_template="<span></span>",
             footer_template=footer if numbered else "<span></span>",
             margin={"top": "18mm", "bottom": "20mm", "left": "18mm", "right": "18mm"} if numbered
             else {"top": "0", "bottom": "0", "left": "0", "right": "0"})
    tmp_html.unlink()


def find_pages(pdf: Path, n: int) -> dict[int, int]:
    from pypdf import PdfReader

    found: dict[int, int] = {}
    for page_no, page in enumerate(PdfReader(str(pdf)).pages, start=1):
        text = page.extract_text() or ""
        for m in re.finditer(r"ZQX(\d{4})QXZ", text):
            found.setdefault(int(m.group(1)), page_no)
    missing = n - len(found)
    if missing:
        print(f"  attenzione: {missing} titoli non trovati nel PDF (indice senza numero)")
    return found


def main() -> None:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader, PdfWriter

    print(f"Capitoli: {len(chapters())}")
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page()
        body_pdf, cover_pdf = Path(tmp) / "body.pdf", Path(tmp) / "cover.pdf"

        body, headings = build_body(None)
        render(page, page_html(body), body_pdf, numbered=True)          # 1a passata
        pages = find_pages(body_pdf, len(headings))
        body, headings = build_body(pages)
        render(page, page_html(body), body_pdf, numbered=True)          # 2a passata
        render(page, cover_html(), cover_pdf, numbered=False)
        browser.close()

        writer = PdfWriter()
        writer.append(str(cover_pdf))
        writer.append(str(body_pdf))
        # Segnalibri: capitoli (h1) con le sezioni (h2) annidate. +1 per la copertina.
        parent = None
        for idx, level, text, _ in headings:
            if idx not in pages:
                continue
            target = pages[idx]  # numero di pagina del corpo = pagina del PDF finale (copertina = 0)
            if level == 1:
                parent = writer.add_outline_item(text, target)
            else:
                writer.add_outline_item(text, target, parent=parent)
        writer.add_metadata({"/Title": f"{TITLE} — guida", "/Author": "Andrea Vagnoli"})
        writer.write(str(OUTPUT))
    n_pages = len(PdfReader(str(OUTPUT)).pages)
    print(f"Guida scritta in {OUTPUT.relative_to(ROOT)} ({n_pages} pagine)")


if __name__ == "__main__":
    main()
