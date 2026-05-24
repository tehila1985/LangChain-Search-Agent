import os
import httpx
import requests
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

def tavily_search_tool(query):
    url = "https://api.tavily.com/search"
    payload = {"api_key": os.getenv("TAVILY_API_KEY"), "query": query, "search_depth": "advanced"}
    response = requests.post(url, json=payload, verify=False)
    return str(response.json())

@tool
def tavily_search(query: str) -> str:
    """Search the internet for information on a given topic."""
    return tavily_search_tool(query)

tools = [tavily_search]

llm = ChatCohere(model="command-r-plus-08-2024")
llm.client.v2._raw_client._client_wrapper.httpx_client.httpx_client = httpx.Client(verify=False)

SYSTEM_PROMPT = """You are a research agent. Your job is to collect information sources from the internet on the topic the user requests.
Run several diverse searches to cover the topic from different angles.
Return a list of relevant sources with a title, link, and short summary for each."""

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

response = agent.invoke({"messages": [{"role": "user", "content": "Tell me about the advantages of NotebookLM"}]})
print(response["messages"][-1].content)