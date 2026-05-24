import os
import ssl
import httpx
import requests
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# הכלי שלנו (נשאר אותו דבר)
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

# הגדרת המוח (LLM) עם Cohere — SSL disabled בגלל בעיית certificates
llm = ChatCohere(model="command-r-plus-08-2024")
llm.client.v2._raw_client._client_wrapper.httpx_client.httpx_client = httpx.Client(verify=False)

# יצירת ה-Agent
agent = create_agent(llm, tools)

# הרצה
response = agent.invoke({"messages": [{"role": "user", "content": "ספרי לי על היתרונות של NotebookLM"}]})
print(response["messages"][-1].content)