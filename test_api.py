import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys from .env file
load_dotenv()

# Check if keys exist
print("🔍 Checking API keys...")
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit()
if not os.getenv("TAVILY_API_KEY"):
    print("⚠️ Warning: TAVILY_API_KEY not found (we will need it later)")

print("✅ API keys found successfully!")

# Test connection with Gemini
print("\n🤖 Testing AI connection (Gemini)...")
try:
    # Using gemini-2.0-flash (fast and free tier)
    model = init_chat_model("gemini-3.8-flash", model_provider="google_genai")
    
    response = model.invoke("I am testing my system. Reply with exactly: 'System is working successfully.'")
    
    print("\n🎉 AI Response:")
    print(response.content)
    print("\n🚀 Excellent! The environment is fully ready to start building.")

except Exception as e:
    print(f"\n❌ Error during connection: {e}")