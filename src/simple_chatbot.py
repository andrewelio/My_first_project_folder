# src/simple_chatbot.py
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise and clear."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm

# File for persistent history
HISTORY_FILE = "chat_history.json"

# Store session histories in memory
_store: dict[str, InMemoryChatMessageHistory] = {}

def load_history_from_file(session_id: str) -> InMemoryChatMessageHistory:
    history = InMemoryChatMessageHistory()
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            for msg in data.get(session_id, []):
                history.add_message(msg)
    return history

def save_history_to_file(session_id: str):
    # dump all sessions (_store) into JSON
    all_data = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {}
    all_data[session_id] = _store[session_id].messages
    with open(HISTORY_FILE, "w") as f:
        json.dump(all_data, f, indent=2, default=lambda o: o.__dict__)

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = load_history_from_file(session_id)
    return _store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

SESSION_ID = "persistent-cli-session"

print("🤖 Chatbot with persistent memory is ready!")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    user = input("You: ")
    if user.lower() in {"quit", "exit"}:
        print("Chatbot: Goodbye! 👋")
        save_history_to_file(SESSION_ID)
        break

    result = chat_with_memory.invoke(
        {"input": user},
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print("Chatbot:", result.content)
    # Save after each turn
    save_history_to_file(SESSION_ID)



