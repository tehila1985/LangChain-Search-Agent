import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch # הייבוא החדש

load_dotenv()

# הגדרת ה-Tool המעודכן
search = TavilySearch(max_results=2)

# בדיקה נוספת שהכל תקין
result = search.invoke("NotebookLM features")
print(result)