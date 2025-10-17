# list_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("🔍 Fetching available models...\n")

try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Model: {m.name} | Description: {m.description}")
except Exception as e:
    print("❌ Error listing models:", e)