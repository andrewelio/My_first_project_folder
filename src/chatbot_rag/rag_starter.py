# rag_starter.py — RAG engine with robust PDF loaders, persistent index,
# hybrid retrieval (Chroma + BM25) + reranking, multi-query expansion,
# and extractor-first answers for CRE facts (strict for facts).

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
from collections import Counter
import shutil
import re

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,          # strong general extractor (fitz)
    UnstructuredPDFLoader,  # robust for messy PDFs (needs unstructured[pdf])
    PyPDFLoader,            # basic fallback
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import LongContextReorder
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

# ---------- paths ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"          # put files under data/<PropertyName>/*
PERSIST_DIR = BASE_DIR / "chroma_db"  # persistent vector DB

# ---------- settings ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
TOP_K = 12
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
STRICT_RAG = True   # when a fact is asked and extractor fails → "Not found in docs."

SYSTEM = (
    "You are a factual assistant. Answer using ONLY the provided Context. "
    "If the answer is not explicitly in the Context, reply exactly: 'Not found in docs.' "
    "Be concise and include numbers/units if present."
)
PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="Question: {question}\n\nContext:\n{context}\n\nAnswer:"
)

# ---------- globals (lazy) ----------
_vectordb: Chroma | None = None
_docs = None
_chunks = None
_bm25: BM25Retriever | None = None
_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

# ---------- loaders ----------
def _load_single_pdf(path: Path):
    """Try multiple PDF loaders from strongest to simplest."""
    try:
        return PyMuPDFLoader(str(path)).load()
    except Exception:
        pass
    try:
        return UnstructuredPDFLoader(str(path)).load()
    except Exception:
        pass
    return PyPDFLoader(str(path)).load()

def load_docs_with_property_tags(data_dir: Path):
    """
    Load .txt and .pdf recursively and tag each Document with:
      - property: immediate parent folder (or 'GLOBAL' if directly in /data)
      - source: filename
    """
    docs = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".pdf":
            loaded = _load_single_pdf(path)
        else:
            continue
        prop = path.parent.name if path.parent != data_dir else "GLOBAL"
        for d in loaded:
            d.metadata["property"] = prop
            d.metadata.setdefault("source", path.name)
        docs.extend(loaded)
    return docs

# ---------- index build / load ----------
def _build_index():
    """(Re)build docs → chunks → embeddings → persistent Chroma, BM25, and cache in memory."""
    global _vectordb, _docs, _chunks, _bm25

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs_with_property_tags(DATA_DIR)
    if not docs:
        # seed so first run never fails
        starter = DATA_DIR / "GLOBAL.txt"
        starter.write_text(
            "Silver Court Apartments: 36 units in Elgin, IL. Asking $4,950,000. "
            "NOI $349,979. Cap rate 8.2%. Year built 1972. 28,500 SF. Lot size 1.2 acres. "
            "Occupancy 96%. Average rent $1,245. Taxes $58,300. Expenses $175,000. "
            "Parking: 40 surface spaces.",
            encoding="utf-8",
        )
        docs = load_docs_with_property_tags(DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=str(PERSIST_DIR))
    vectordb.persist()

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    _vectordb, _docs, _chunks, _bm25 = vectordb, docs, chunks, bm25

def _load_or_build():
    """Load persistent index if present; else build it (populate globals)."""
    global _vectordb, _docs, _chunks, _bm25
    if _vectordb is not None and _bm25 is not None and _chunks is not None:
        return
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        _vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
        _docs = load_docs_with_property_tags(DATA_DIR)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        )
        _chunks = splitter.split_documents(_docs)
        _bm25 = BM25Retriever.from_documents(_chunks)
        _bm25.k = TOP_K
    else:
        _build_index()

def rebuild_index_on_demand():
    """Public: wipe and rebuild the persistent index and in-memory handles."""
    global _vectordb, _docs, _chunks, _bm25
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    _vectordb = _docs = _chunks = _bm25 = None
    _build_index()

# ---------- retrieval helpers ----------
def _doc_char_counts(property_name: Optional[str] = None):
    """Return [(source, total_chars)] for current in-memory chunks, optionally filtered by property."""
    _load_or_build()
    items = {}
    for d in _chunks:
        if property_name and d.metadata.get("property") != property_name:
            continue
        key = (d.metadata.get("source", "unknown"), d.metadata.get("property","GLOBAL"))
        items.setdefault(key, 0)
        items[key] += len(d.page_content or "")
    # sort largest first
    return sorted([(src, prop, n) for (src, prop), n in items.items()], key=lambda x: -x[2])

def _years_in_text(s: str) -> list[str]:
    return re.findall(r"\b(20\d{2})\b", s)

def _format_tag(d) -> str:
    src = d.metadata.get("source", "unknown")
    pg = d.metadata.get("page")
    prop = d.metadata.get("property", "GLOBAL")
    where = f"{src} p.{pg+1}" if pg is not None else src
    return f"[{prop} · {where}]"

def _dedupe(docs: List) -> List:
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("property"), d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def _paraphrases(q: str) -> List[str]:
    """Light multi-query expansion for better recall on common intents."""
    ql = q.lower()
    alts = [q]
    if any(k in ql for k in ["how many units", "unit count", "number of units", "units"]):
        alts += ["unit count", "number of units", "units total", "how many apartments"]
    if any(k in ql for k in ["asking price", "ask price", "list price", "purchase price", "offer price", "price"]):
        alts += ["sale price", "purchase price", "list price", "asking price"]
    if "noi" in ql or "net operating income" in ql:
        alts += ["NOI", "net operating income"]
    if "cap rate" in ql or "caprate" in ql or "cap-rate" in ql:
        alts += ["cap rate", "capitalization rate"]
    if "address" in ql or "where is" in ql:
        alts += ["property address", "street address"]
    if any(k in ql for k in ["square feet","sq ft","sf","building size","building sf","rsf","gsf"]):
        alts += ["building size", "building sf", "total square feet"]
    if any(k in ql for k in ["lot","site","parcel","acres","acre"]):
        alts += ["lot size", "site size", "parcel size", "acres"]
    if any(k in ql for k in ["occupancy","occupied","vacancy"]):
        alts += ["occupancy", "occupancy rate", "vacancy rate"]
    if any(k in ql for k in ["average rent","avg rent","rent per unit"]):
        alts += ["average rent", "rent per unit", "avg monthly rent"]
    if "tax" in ql:
        alts += ["real estate taxes", "taxes"]
    if any(k in ql for k in ["expense","expenses","opex"]):
        alts += ["operating expenses", "expenses", "opex"]
    if any(k in ql for k in ["parking","spaces"]):
        alts += ["parking spaces", "number of parking spaces"]
    # de-dupe / keep small
    seen, out = set(), []
    for s in alts:
        if s not in seen:
            out.append(s); seen.add(s)
    return out[:6]

def _retrieve_docs(question: str, property_name: Optional[str]) -> List:
    """Hybrid retrieval: semantic (Chroma) + keyword (BM25) across multi-queries, reranked."""
    _load_or_build()
    queries = _paraphrases(question)

    # semantic hits (filter at query-time if property constrained)
    if property_name and property_name.upper() != "ALL":
        sem_ret = _vectordb.as_retriever(search_kwargs={"k": TOP_K, "filter": {"property": property_name}})
    else:
        sem_ret = _vectordb.as_retriever(search_kwargs={"k": TOP_K})

    sem_docs = []
    for q in queries:
        sem_docs.extend(sem_ret.invoke(q))

    # keyword hits (post-filter by property)
    bm_docs = []
    for q in queries:
        hits = _bm25.invoke(q)
        if property_name and property_name.upper() != "ALL":
            hits = [d for d in hits if d.metadata.get("property") == property_name]
        bm_docs.extend(hits)

    merged = _dedupe(sem_docs + bm_docs)
    reranked = LongContextReorder().transform_documents(merged)
    return reranked[:TOP_K]

# ---------- extractor-first rules ----------
_MONEY = r"\$?\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?"
_PCT   = r"\d+(?:\.\d+)?\s?%"
_SF    = r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\s?(?:sf|sq\.?\s*ft|square\s*feet)\b"
_ACRE  = r"\d+(?:\.\d+)?\s?(?:acre|acres)\b"
_NOISY_NEAR_UNITS = ("photo", "photos", "floor plan", "floor plans", "plan", "plans", "model", "render")

def _extract_noi(context: str, year: str | None) -> str | None:
    # Prefer NOI amounts near the requested year
    if year:
        # NOI ... $X with year nearby
        for m in re.finditer(r"(?:noi|net\s*operating\s*income)[^\n]{0,120}(" + _MONEY + r")",
                             context, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            window = context[max(0, s-80): min(len(context), e+80)]
            if re.search(fr"\b{re.escape(year)}\b", window):
                return m.group(1)

        # YEAR ... NOI ... $X
        m = re.search(fr"\b{re.escape(year)}\b[^\n]{{0,120}}(?:noi|net\s*operating\s*income)[^\n]{{0,120}}("
                      + _MONEY + r")", context, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        # YEAR followed soon by a money figure (fallback)
        m = re.search(fr"\b{re.escape(year)}\b[^\n]{{0,80}}(" + _MONEY + r")", context, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    # Generic NOI (no year specified)
    return _search([rf"(?:noi|net\s*operating\s*income)[:\s]*({_MONEY})"], context)

def _search(patterns: List[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

# --- Smart extractors ---
def _iter_unit_candidates(context: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    pats = [
        r"units?\s*[:\-]\s*(\d{1,4})",
        r"(?:^|\b)(\d{1,4})\s+units\b",
        r"unit\s*count\s*[:\-]\s*(\d{1,4})",
    ]
    for pat in pats:
        for m in re.finditer(pat, context, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            window = context[max(0, s-80): min(len(context), e+80)].lower()
            if any(k in window for k in _NOISY_NEAR_UNITS):
                continue
            try:
                val = int(m.group(1).replace(",", ""))
                if 0 < val < 2000:
                    out.append((val, window))
            except Exception:
                pass
    return out

def _extract_units(context: str) -> str | None:
    candidates = _iter_unit_candidates(context)
    if not candidates:
        return None
    good = ["unit mix","mix","property highlights","highlights","property information",
            "information","property summary","summary","details","specs","specifications",
            "overview","executive summary","investment highlights"]
    score = Counter()
    for val, window in candidates:
        w = 1 + (2 if any(lbl in window for lbl in good) else 0)
        score[val] += w
    best_val, _ = score.most_common(1)[0]
    return str(best_val)

def _extract_building_sf(context: str) -> str | None:
    """
    Choose the building's total SF, not per-unit SF from a rent roll.
    Score by label proximity, penalize rent-roll noise, slight boost for larger numbers.
    """
    building_labels = (
        "building size", "building area", "gross building area", "gba",
        "building sf", "property information", "offering summary", "property summary"
    )
    noisy_unit_context = (
        "rent roll", "suite", "bedrooms", "bathrooms", "size sf",
        "lease start", "lease end", "rent / sf", "unit", "units"
    )

    sf_matches = []
    for m in re.finditer(_SF, context, flags=re.IGNORECASE):
        s, e = m.start(), m.end()
        val_text = m.group(0)
        window = context[max(0, s-80): min(len(context), e+80)].lower()
        num = None
        try:
            num = float(re.sub(r"[^\d.]", "", val_text.replace(",", "")))
        except Exception:
            pass
        sf_matches.append((val_text, window, num))

    if not sf_matches:
        return None

    best, best_score = None, -1e9
    for val_text, window, num in sf_matches:
        score = 0
        if any(lbl in window for lbl in building_labels):
            score += 5
        if "total" in window or "totals" in window:
            score += 2
        if any(noise in window for noise in noisy_unit_context):
            score -= 5
        if num is not None:
            score += min(num / 5000.0, 4.0)  # prefer bigger numbers, capped
        if score > best_score:
            best_score, best = score, val_text
    return best

def _extract_lot_size(context: str) -> str | None:
    """Prefer acres labeled as lot/site/parcel; fallback to big SF labeled as lot/site."""
    lot_labels = ("lot size", "site size", "parcel size", "lot area", "site area", "parcel area")
    # Try acres first
    best, best_score = None, -1e9
    for m in re.finditer(_ACRE, context, flags=re.IGNORECASE):
        s, e = m.start(), m.end()
        val = m.group(0)
        window = context[max(0, s-60): min(len(context), e+60)].lower()
        score = 1 + (3 if any(lbl in window for lbl in lot_labels) else 0)
        if score > best_score:
            best, best_score = val, score
    if best:
        return best
    # Fallback to SF with lot labels, prefer larger
    for m in re.finditer(_SF, context, flags=re.IGNORECASE):
        s, e = m.start(), m.end()
        val = m.group(0)
        window = context[max(0, s-80): min(len(context), e+80)].lower()
        if any(lbl in window for lbl in lot_labels):
            return val
    return None

def _extract_avg_rent(context: str) -> str | None:
    """Prefer explicit 'average/avg rent' or 'rent per unit' lines; avoid table noise when possible."""
    out = _search([
        rf"(?:average|avg)\s*rent[:\s\-]*({_MONEY})",
        rf"rent\s*per\s*unit[:\s\-]*({_MONEY})",
    ], context)
    if out:
        return out
    # Fallback: if one consistent per-unit rent appears repeatedly, take mode
    rents = []
    for m in re.finditer(_MONEY, context):
        s, e = m.start(), m.end()
        window = context[max(0, s-30): min(len(context), e+30)].lower()
        if "rent" in window and "/ sf" not in window:
            rents.append(m.group(0))
    if rents:
        return Counter(rents).most_common(1)[0][0]
    return None

def _extract_value(question: str, context: str) -> str | None:
    """Deterministically extract common CRE fields from retrieved context."""
    q = question.lower()

    # Price
    if any(k in q for k in ["ask price","asking price","purchase price","offer price","list price","price"]):
        return _search([
            rf"(?:asking|ask|list|purchase|offer)\s*price[:\s]*({_MONEY})",
            rf"price[:\s]*({_MONEY})",
            rf"({_MONEY})\s*(?:asking|ask|list)\s*price",
        ], context)

    # NOI
    if "noi" in q or "net operating income" in q:
        return _search([rf"(?:noi|net\s*operating\s*income)[:\s]*({_MONEY})"], context)

    # Cap rate
    if "cap rate" in q or "caprate" in q or "cap-rate" in q:
        return _search([
            rf"(?:cap\s*rate|cap\-?rate|caprate)[:\s]*({_PCT})",
            rf"({_PCT})\s*(?:cap\s*rate|cap\-?rate|caprate)",
        ], context)

    # Units
    if any(k in q for k in ["how many units","units","unit count","number of units"]):
        u = _extract_units(context)
        if u: return u

    # Address
    if "address" in q or "property address" in q or "where is" in q:
        addr = _search([r"(?:address|property\s*address)\s*[:\-]\s*([^\n]+)"], context)
        if addr: return addr
        m = re.search(
            r"(^|\n)\s*(\d{2,6}\s+[A-Za-z0-9\.\- ]+)\s*,?\s*([A-Za-z\.\- ]+),\s*([A-Z]{2})\s*\d{5}(-\d{4})?",
            context, flags=re.IGNORECASE)
        if m:
            return f"{m.group(2).strip()}, {m.group(3).strip()}, {m.group(4).strip()}"

    # Year Built / Renovated
    if "year built" in q or ("built" in q and "year" in q):
        return _search([r"(?:year\s*built)[:\s\-]*([12]\d{3})", r"built\s*(?:in)?\s*([12]\d{3})"], context)
    if "year renovated" in q or "renovated" in q:
        return _search([r"(?:year\s*renovated|renovated)[:\s\-]*([12]\d{3})"], context)

    # Building SF (smart)
    if any(k in q for k in ["size","square feet","sq ft","sf","building sf","building size","rsf","gsf"]):
        sf = _extract_building_sf(context)
        if sf: return sf

    # Lot size (smart)
    if "lot" in q or "site" in q or "parcel" in q:
        lot = _extract_lot_size(context)
        if lot: return lot

    # Occupancy / Vacancy
    if "occupancy" in q or "occupied" in q or "vacancy" in q:
        occ = _search([rf"(?:occupancy|occupied)[:\s\-]*({_PCT})"], context)
        if occ: return occ
        vac = _search([rf"(?:vacancy)[:\s\-]*({_PCT})"], context)
        if vac:
            try:
                val = float(vac.replace('%','').strip())
                return f"{max(0.0, 100.0 - val):.1f}%"
            except Exception:
                return vac

    # Average rent (smart)
    if "average rent" in q or "avg rent" in q or "rent per unit" in q:
        r = _extract_avg_rent(context)
        if r: return r

    # Taxes / Expenses
    if "tax" in q:
        return _search([rf"(?:real\s*estate\s*)?tax(?:es)?[:\s\-]*({_MONEY})"], context)
    if "expense" in q or "opex" in q:
        return _search([rf"(?:expenses|operating\s*expenses|opex)[:\s\-]*({_MONEY})"], context)

    # Parking
    if "parking" in q or "spaces" in q:
        out = _search([
            r"parking[:\s\-]*([0-9,]+)\s*(?:spaces|stalls)",
            r"([0-9,]+)\s*(?:parking|spaces|stalls)",
        ], context)
        if out: return out

    # Unit mix
    if "unit mix" in q or "bedroom mix" in q or "mix" in q:
        m = re.search(r"((?:\d+\s*x\s*[0-9\-]+(?:br|bed)?(?:\s*,\s*)?)+)", context, flags=re.IGNORECASE)
        if m: return m.group(1)

    return None

def _field_detected(q: str) -> bool:
    q = q.lower()
    keys = [
        "ask price","asking price","purchase price","offer price","list price","price",
        "noi","net operating income",
        "cap rate","cap-rate","caprate",
        "how many units","units","unit count","number of units",
        "address","property address","where is",
        "year built","built","year renovated","renovated",
        "square feet","sq ft","sf","building size","building sf","rsf","gsf",
        "lot","site","parcel","acre","acres",
        "occupancy","occupied","vacancy",
        "average rent","avg rent","rent per unit",
        "tax","taxes","expense","expenses","opex",
        "parking","spaces","unit mix","bedroom mix","mix",
    ]
    return any(k in q for k in keys)

# ---------- answer ----------
def ask_rag(question: str, property_name: Optional[str] = None) -> str:
    hits = _retrieve_docs(question, property_name)
    context = "\n\n".join(f"{_format_tag(d)}\n{d.page_content}" for d in hits)

    # 1) deterministic extraction
    extracted = _extract_value(question, context)
    if extracted:
        return extracted

    # 2) strict mode for fact asks
    if STRICT_RAG and _field_detected(question):
        return "Not found in docs."

    # 3) grounded LLM answer (for summaries/open questions)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=PROMPT.format(question=question, context=context)),
    ]
    return _llm.invoke(msgs).content

# ---------- debugging ----------
def debug_hits(question: str, property_name: Optional[str] = None) -> None:
    hits = _retrieve_docs(question, property_name)
    print(f"[debug] hits: {len(hits)}")
    for i, d in enumerate(hits, 1):
        print(f"\n--- Hit {i}: {_format_tag(d)} ---\n{d.page_content[:800]}...\n")

    # Extra debug for unit questions: show candidates and scores
    low_q = question.lower()
    if any(k in low_q for k in ["how many units", "units", "unit count", "number of units"]):
        text = "\n\n".join(d.page_content for d in hits)
        cands = _iter_unit_candidates(text)
        if not cands:
            print("[debug units] no candidates found.")
        else:
            good = ["unit mix","mix","property highlights","highlights","property information",
                    "information","property summary","summary","details","specs","specifications",
                    "overview","executive summary","investment highlights"]
            score = Counter()
            for val, window in cands:
                w = 1 + (2 if any(lbl in window for lbl in good) else 0)
                score[val] += w
            print("[debug units] candidates:", [v for v, _ in cands])
            print("[debug units] scores:", dict(score))
            best, _ = score.most_common(1)[0]
            print("[debug units] chosen:", best)

# ---------- simple CLI (optional) ----------
if __name__ == "__main__":
    _load_or_build()
    active = "ALL"
    print(f"Index ready from: {DATA_DIR.resolve()}")
    print("Commands: 'list' · 'use <PropertyName>' · 'reindex' · 'exit'\n")
    while True:
        q = input(f"[{active}] Ask about your docs: ").strip()
        if not q:
            continue
        low = q.lower()
        if low in {"exit", "quit", "q"}:
            break
        if low == "list":
            props = sorted({d.metadata.get("property", "GLOBAL") for d in _chunks})
            print("Properties:", ", ".join(props), "\n"); continue
        if low.startswith("use "):
            active = q[4:].strip() or "ALL"
            print(f"[router] Active property set to: {active}\n"); continue
        if low == "reindex":
            rebuild_index_on_demand()
            print("[index] Rebuilt from /data.\n"); continue
        if low == "probe":
            rows = _doc_char_counts(active if active.upper() != "ALL" else None)
            print("Extracted text by file (chars):")
            for src, prop, n in rows:
                print(f"  [{prop}] {src:40s}  {n}")
            print()
            continue

        print(ask_rag(q, property_name=active), "\n")
