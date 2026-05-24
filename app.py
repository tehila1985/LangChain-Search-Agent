import os
import httpx
import requests
import operator
import urllib3
import streamlit as st
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

st.set_page_config(page_title="NotebookLM בקטנה", page_icon="📚", layout="centered")
st.title("📚 NotebookLM בקטנה")
st.caption("הכנס נושא — ה-Agent יחפש מקורות, אתה תאשר, והוא יסכם.")

# --- LLM ---
@st.cache_resource
def get_llm():
    llm = ChatCohere(model="command-r-plus-08-2024")
    llm.client.v2._raw_client._client_wrapper.httpx_client.httpx_client = httpx.Client(verify=False)
    return llm

llm = get_llm()

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    sources: List[str]
    final_summary: str

# --- Nodes ---
def agent_node(state):
    query = state["messages"][-1]
    url = "https://api.tavily.com/search"
    sources = []
    for search_query in [query, f"{query} יתרונות", f"{query} סקירה"]:
        payload = {"api_key": os.getenv("TAVILY_API_KEY"), "query": search_query, "search_depth": "advanced"}
        res = requests.post(url, json=payload, verify=False)
        for r in res.json().get("results", []):
            sources.append({"title": r["title"], "url": r["url"], "content": r.get("content", "")})
    return {"sources": sources}

def summarizer_node(state):
    approved = state.get("sources", [])
    sources_text = "\n\n".join([f"{s['title']}\n{s['content']}" for s in approved])
    prompt = f"""להלן מקורות מידע שנאספו מהאינטרנט.
כתוב סיכום מקצועי בעברית בפסקאות קצרות וקריאות.
הסיכום יכיל כותרת פתיחה, אחריה פסקאות רגילות, ובמקומות רלוונטיים בלבד הוסף נקודה אחת או שתיים עם אימוג'י מתאים.
אל תגזים בשימוש באימוג'ים או בנקודות.

{sources_text}"""
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

# --- Graph ---
@st.cache_resource
def build_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "summarizer")
    workflow.add_edge("summarizer", END)
    return workflow.compile(checkpointer=MemorySaver(), interrupt_after=["agent"])

graph = build_app()

# --- Session State ---
for key, default in [("stage", "input"), ("sources", []), ("config", None), ("summary", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Stage: input ---
if st.session_state.stage == "input":
    topic = st.text_input("🔍 על מה לחפש?", placeholder="לדוגמה: בינה מלאכותית בחינוך")
    if st.button("חפש מקורות", type="primary") and topic:
        with st.spinner("מחפש מקורות..."):
            config = {"configurable": {"thread_id": topic}}
            st.session_state.config = config
            for event in graph.stream({"messages": [topic]}, config=config):
                if "agent" in event:
                    st.session_state.sources = event["agent"]["sources"]
        st.session_state.stage = "review"
        st.rerun()

# --- Stage: review ---
elif st.session_state.stage == "review":
    st.subheader("📋 בחר את המקורות שברצונך לכלול בסיכום")
    selected = []
    for i, source in enumerate(st.session_state.sources):
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            checked = st.checkbox("", value=True, key=f"src_{i}")
        with col2:
            st.markdown(f"**{source['title']}**  \n[{source['url']}]({source['url']})")
            with st.expander("תצוגה מקדימה"):
                st.write(source["content"][:300] + "...")
        if checked:
            selected.append(source)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ אשר וסכם", type="primary"):
            with st.spinner("יוצר סיכום..."):
                state_snapshot = graph.get_state(st.session_state.config)
                graph.update_state(st.session_state.config, {"sources": selected})
                for event in graph.stream(None, st.session_state.config):
                    if "summarizer" in event:
                        st.session_state.summary = event["summarizer"]["final_summary"]
            st.session_state.stage = "summary"
            st.rerun()
    with col_b:
        if st.button("🔄 חיפוש חדש"):
            for key in ["stage", "sources", "config", "summary"]:
                del st.session_state[key]
            st.rerun()

# --- Stage: summary ---
elif st.session_state.stage == "summary":
    st.subheader("📝 סיכום")
    st.markdown(
        f"<div style='direction:rtl; text-align:right; line-height:2'>{st.session_state.summary.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )
    if st.button("🔄 חיפוש חדש"):
        for key in ["stage", "sources", "config", "summary"]:
            del st.session_state[key]
        st.rerun()
