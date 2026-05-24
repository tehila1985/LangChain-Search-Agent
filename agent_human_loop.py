import os
import httpx
import requests
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    sources: List[str]
    final_summary: str

llm = ChatCohere(model="command-r-plus-08-2024")
llm.client.v2._raw_client._client_wrapper.httpx_client.httpx_client = httpx.Client(verify=False)

def agent_node(state):
    query = state['messages'][-1]
    url = "https://api.tavily.com/search"
    sources = []
    for search_query in [query, f"{query} יתרונות", f"{query} סקירה"]:
        payload = {"api_key": os.getenv("TAVILY_API_KEY"), "query": search_query, "search_depth": "advanced"}
        res = requests.post(url, json=payload, verify=False)
        for r in res.json().get("results", []):
            sources.append(f"{r['title']} | {r['url']}\n{r.get('content', '')}")
    return {"sources": sources}

def summarizer_node(state):
    sources = state.get('sources', [])
    sources_text = "\n".join(sources)
    prompt = f"להלן מקורות מידע שנאספו מהאינטרנט. אנא כתוב סיכום קצר, מקצועי וקולע בעברית על בסיסם:\n\n{sources_text}"
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", "summarizer")
workflow.add_edge("summarizer", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_after=["agent"])

config = {"configurable": {"thread_id": "1"}}
initial_input = {"messages": ["ספרי לי על היתרונות של NotebookLM"]}

for event in app.stream(initial_input, config=config):
    pass

user_choice = input("\nהמקורות נמצאו. האם לאשר אותם ולעבור לסיכום? (כן/לא): ")

if user_choice.strip().lower() == "כן":
    print("יוצר סיכום סופי...")
    for event in app.stream(None, config=config):
        if "summarizer" in event:
            print("\n--- סיכום סופי ---")
            print(event["summarizer"]["final_summary"])
else:
    print("התהליך נעצר.")