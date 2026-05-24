import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")

print("Running raw search...")

# קריאה ישירה ל-API של Tavily תוך עקיפת בדיקת SSL
url = "https://api.tavily.com/search"
payload = {
    "api_key": api_key,
    "query": "NotebookLM features",
    "search_depth": "advanced"
}

try:
    # כאן ה-verify=False עושה את העבודה
    response = requests.post(url, json=payload, verify=False)
    print("Status Code:", response.status_code)
    print(response.json())
except Exception as e:
    print(f"Error: {e}")