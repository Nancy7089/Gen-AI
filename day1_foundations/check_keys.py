
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

print("=== API KEY STATUS CHECK ===")

# 1. Google / Gemini
google_key = os.getenv("GOOGLE_API_KEY")

if not google_key or google_key == "your_gemini_api_key_here":
    print("❌ GOOGLE_API_KEY: Missing or placeholder")
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0.0
        )
        llm.invoke("Reply with OK")
        print(f"✅ GOOGLE_API_KEY: Working (...{google_key[-4:]})")
    except Exception as e:
        print(f"❌ GOOGLE_API_KEY: Failed -> {e}")

# 2. Groq
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key or groq_key == "your_groq_api_key_here":
    print("⚠️ GROQ_API_KEY: Missing or placeholder")
else:
    try:
        from langchain_groq import ChatGroq

        groq_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=groq_key
        )
        groq_llm.invoke("Reply with OK")
        print(f"✅ GROQ_API_KEY: Working (...{groq_key[-4:]})")
    except Exception as e:
        print(f"❌ GROQ_API_KEY: Failed -> {e}")

# 3. Tavily
tavily_key = os.getenv("TAVILY_API_KEY")

if not tavily_key or tavily_key == "your_tavily_api_key_here":
    print("⚠️ TAVILY_API_KEY: Missing or placeholder")
else:
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)
        client.search(query="test", max_results=1)
        print(f"✅ TAVILY_API_KEY: Working (...{tavily_key[-4:]})")
    except Exception as e:
        print(f"❌ TAVILY_API_KEY: Failed -> {e}")

# 4. OpenWeather
weather_key = os.getenv("OPENWEATHER_API_KEY")

if not weather_key or weather_key == "your_openweather_api_key_here":
    print("⚠️ OPENWEATHER_API_KEY: Missing or placeholder")
else:
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        response = requests.get(
            url,
            params={"q": "Pune", "appid": weather_key},
            timeout=15
        )

        if response.status_code == 200:
            print(f"✅ OPENWEATHER_API_KEY: Working (...{weather_key[-4:]})")
        else:
            message = response.json().get("message", response.text)
            print(
                f"❌ OPENWEATHER_API_KEY: "
                f"HTTP {response.status_code} -> {message}"
            )
    except Exception as e:
        print(f"❌ OPENWEATHER_API_KEY: Failed -> {e}")
