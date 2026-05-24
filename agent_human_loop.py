import os
import httpx
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# הגדרת ה-State
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    sources: List[str]
    final_summary: str # הוספנו מקום לסיכום הסופי

# ה-LLM המשותף
llm = ChatCohere(model="command-r-plus-08-2024")
llm.client.v2._raw_client._client_wrapper.httpx_client.httpx_client = httpx.Client(verify=False)

# 1. ה-Node של החיפוש
def agent_node(state):
    query = state['messages'][-1]
    # כאן השתמשנו ב-llm ככלי עזר לחיפוש אם צריך, או פשוט העברנו את המקורות
    # לצרכי הפשטות, נניח שהמקורות כבר הגיעו מ-Tavily
    return {"sources": ["מקור 1: Google NotebookLM...", "מקור 2: NotebookLM Enterprise..."]}

def summarizer_node(state):
    # כאן אנחנו מוודאות שהוא שואב את המקורות מה-State ששמרנו בשלב הקודם
    sources = state.get('sources', [])
    
    # בואי נהפוך את רשימת המקורות לטקסט ארוך שה-LLM יבין
    sources_text = "\n".join(sources) 
    
    prompt = f"להלן מקורות מידע שנאספו מהאינטרנט. אנא כתוב סיכום קצר, מקצועי וקולע בעברית על בסיסם:\n\n{sources_text}"
    
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

# בניית הגרף
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("summarizer", summarizer_node)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "summarizer")
workflow.add_edge("summarizer", END)

# קומפילציה עם עצירה אחרי ה-agent
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_after=["agent"])

# הרצה
config = {"configurable": {"thread_id": "1"}}
initial_input = {"messages": ["ספרי לי על היתרונות של NotebookLM"]}

# מריצים עד העצירה
for event in app.stream(initial_input, config=config):
    pass

# אישור אנושי
user_choice = input("\nהמקורות נמצאו. האם לאשר אותם ולעבור לסיכום? (כן/לא): ")

if user_choice.strip().lower() == "כן":
    print("יוצר סיכום סופי...")
    for event in app.stream(None, config=config):
        if "summarizer" in event:
            print("\n--- סיכום סופי ---")
            print(event["summarizer"]["final_summary"])
else:
    print("התהליך נעצר.")