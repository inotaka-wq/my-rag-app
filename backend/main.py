import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # CORSを許可するために追加
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# CORS設定: Reactアプリからのアクセスを許可する
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ReactアプリのURL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


model = genai.GenerativeModel('gemini-1.5-flash')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
async def ask(query: Query):
    # ... (RAGのロジックはここに実装) ...
    prompt = f"質問: {query.question}\n\nこの質問に簡潔に答えてください。"
    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"error": str(e)}
