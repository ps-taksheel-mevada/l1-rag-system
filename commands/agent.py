import os
from langchain_classic.agents import create_react_agent, AgentExecutor, Tool
from langchain_ollama import ChatOllama
from commands.init_db import vector_store
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0,
    # base_url="http://localhost:11434"
)

def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        str: Formatted search results with document titles and content
    """
    results = vector_store.similarity_search(query, k=1)
    if not results:
        return "No relevant documents found in the knowledge base."
    
    formatted_results = []
    for result in results:
        title = result.metadata.get('title', 'Unknown')
        url = result.metadata.get('source', '')
        content = result.page_content
        formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
    
    return "\n\n".join(formatted_results)

tools = [
    Tool(
        name="Knowledge Base Search",
        func=search_knowledge_base,
        description=(
            """Use this tool to search the knowledge base for relevant information.
            Pass your search query as the input.
            Example: 'What are the benefits of machine learning?'
            Returns relevant documents with titles, content, and sources."""
        )
    )
]

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: A clear and concise answer to the question with sources cited (Full source URL)

Guidelines:
- Search the knowledge base if you need information to answer the question
- If the knowledge base contains relevant information, provide a brief, direct response
- If the answer is not found or you are uncertain, respond: "I don't know"
- Always cite your sources showing the document title
- Don't Ask knowladge base same questions again.
- If user greets you or just start "hello" or "how are you" or anything like that you just have to greet user without calling any tool.
- do not answer questions that are not related to the knowledge base, just respond with "I don't know" or greet user if it's a greeting.
Begin!

Question: {input}
Chat History: {chat_history}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)