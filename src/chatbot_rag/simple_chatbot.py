# simple_chatbot.py — single entry chatbot with auto-RAG + property routing + reindex

from __future__ import annotations
from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# RAG engine (your rag_starter.py)
from rag_starter import (
    ask_rag,
    rebuild_index_on_demand,
    debug_hits,           # optional: for 'why ...' debugging
)

load_dotenv()

# ---------------- Core LLM (normal chat) ----------------
LLM_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

SYSTEM_BEHAVIOR = (
    "You are a concise, accurate assistant for Samaritan Properties. "
    "Use plain language and keep answers tight unless asked for detail."
)

base_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BEHAVIOR),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# minimal in-memory chat history (you can persist to JSON later)
history: list[dict] = []

def normal_llm_reply(user_text: str) -> str:
    msgs = base_prompt.format_messages(history=history, input=user_text)
    return llm.invoke(msgs).content


# ---------------- RAG routing ----------------
DOC_KEYWORDS = {
    # CRE & document terms (expand as you like)
    "om","offering memorandum","rent roll","t12","p&l","loi","lease","cap rate",
    "noi","underwrite","underwriting","unit mix","brochure","prospectus",
    "address","sq ft","tenant","lease term","cam","nnn",
    # price phrasing
    "ask price","asking price","price","$","purchase price","offer price","list price",
}

def looks_like_docs_question(text: str) -> bool:
    t = text.lower()
    if t.startswith("docs:"):
        return True
    return any(k in t for k in DOC_KEYWORDS)


# ---------------- Property context + index mgmt ----------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
ACTIVE_PROPERTY = "ALL"   # default – search across all properties

def list_property_names() -> list[str]:
    """
    Lists property 'contexts' based on subfolders under /data.
    If files are directly under /data, they'll be tagged GLOBAL by rag_starter.
    """
    names = set()
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.is_dir():
                names.add(p.name)
    # Always allow ALL and (implicitly) GLOBAL for un-nested files
    return sorted(names)

def rebuild_index():
    """Calls rag_starter to wipe + rebuild persistent index and in-memory handles."""
    rebuild_index_on_demand()

# ---------------- REPL ----------------
if __name__ == "__main__":
    print("💬 Chatbot ready. Type normally. Prefix with 'docs:' to force document-grounded answers.")
    print("Commands:")
    print("  use <PropertyName>   → switch active property context (folder under /data)")
    print("  list                 → list available property folders")
    print("  reindex              → rebuild the embeddings from files under /data")
    print("  why <your query>     → show retrieved chunks for debugging")
    print("  exit                 → quit\n")

    while True:
        user = input(f"[{ACTIVE_PROPERTY}] You: ").strip()
        if not user:
            continue

        low = user.lower()

        # ----- commands -----
        if low in {"exit", "quit", "q"}:
            break

        if low == "list":
            props = list_property_names()
            if props:
                print("Properties:", ", ".join(props), "\n")
            else:
                print("No subfolders found under /data. Using ALL/GLOBAL.\n")
            continue

        if low.startswith("use "):
            chosen = user[4:].strip()
            if not chosen:
                print("Usage: use <PropertyName>\n")
                continue
            # We allow any name; if it doesn't exist, RAG will just return 'Not found in docs.' as needed
            ACTIVE_PROPERTY = chosen
            print(f"[router] Active property set to: {ACTIVE_PROPERTY}\n")
            continue

        if low == "reindex":
            rebuild_index()
            print("[index] Rebuilt from files under /data. You can 'use <PropertyName>' and ask again.\n")
            continue

        if low.startswith("why "):
            q = user[4:].strip()
            try:
                print("[debug] Top retrieved chunks:")
                debug_hits(q, ACTIVE_PROPERTY)  # prints chunks to console
                print()
            except Exception as e:
                print(f"[debug error] {e}\n")
            continue

        # ----- chat logic -----
        try:
            if looks_like_docs_question(user):
                q = user[5:].strip() if low.startswith("docs:") else user
                answer = ask_rag(q, property_name=ACTIVE_PROPERTY)
            else:
                answer = normal_llm_reply(user)
        except Exception as e:
            answer = f"[error] {e}"

        print(f"Bot: {answer}\n")

        # save minimal history for normal chat continuity (RAG answers don't need to go into the next prompt)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": answer})
