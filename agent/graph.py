import sqlite3
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma

from agent.tools import create_tools

SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base via the 'retrieve' tool.

Guidelines:
- Use the 'retrieve' tool to search the knowledge base for factual information before answering.
- Provide concise, accurate responses based on retrieved documents.
- Cite your sources by including the source URL or title in your answer.
- If the retrieved information doesn't answer the question, say "I don't have enough information to answer that."
- For general conversation (greetings, how are you, etc.), respond naturally without calling any tools."""


def build_graph(
    llm: BaseChatModel,
    vector_store: Chroma,
    checkpoint_db: str = "checkpoints.db",
):
    tools = create_tools(vector_store)
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
        checkpointer=checkpointer,
    )

    return graph, tools
