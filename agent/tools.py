from langchain_core.tools import tool
from langchain_chroma import Chroma


def create_tools(vector_store: Chroma) -> list:
    @tool
    def retrieve(query: str) -> str:
        """Retrieve relevant information from the knowledge base to answer the user's query.
        Always use this tool when you need factual information from the indexed documents."""
        docs = vector_store.similarity_search(query, k=4)

        if not docs:
            return "No relevant documents found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown source")
            title = doc.metadata.get("title", "")
            header = f"{title} ({source})" if title else source
            results.append(f"[{i}] Source: {header}\n{doc.page_content}\n")

        return "\n".join(results)

    return [retrieve]
