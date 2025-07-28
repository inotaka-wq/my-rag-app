# backend/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChainとFAISS関連のライブラリをインポート
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 環境変数の読み込みとAPIキーの設定
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- グローバル変数 ---
# ベクトルDBをグローバル変数として保持
vector_store = None

# --- FastAPIアプリの初期化 ---
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAGのセットアップ（起動時に一度だけ実行） ---


@app.on_event("startup")
def setup_rag():
    global vector_store
    print("サーバー起動時にRAGのセットアップを開始します...")

    # 1. ドキュメントの読み込み
    # 'backend/docs' ディレクトリ内のPDFをすべて読み込む
    loader = DirectoryLoader(
        './docs',                # ドキュメントがあるディレクトリ
        glob="**/*.pdf",         # PDFファイルのみを対象
        loader_cls=PyPDFLoader   # PDFを読み込むクラスを指定
    )
    documents = loader.load()

    if not documents:
        print("ドキュメントが見つかりませんでした。'backend/docs'にPDFファイルを置いてください。")
        return

    # 2. ドキュメントの分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. ベクトル化（Embedding）モデルの準備
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. FAISSによるベクトルDBの作成
    # 読み込んだドキュメントをベクトル化し、FAISSインデックスをメモリ上に作成
    vector_store = FAISS.from_documents(texts, embeddings)
    print("RAGのセットアップが完了しました。")


# --- APIエンドポイント ---
class Query(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "RAG API is running."}


@app.post("/ask")
async def ask(query: Query):
    global vector_store
    if vector_store is None:
        return {"error": "ベクトルストアが準備できていません。ドキュメントがあるか確認してください。"}

    # 1. 質問に関連する文書をベクトルDBから検索
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 上位3件を検索
    retrieved_docs = retriever.invoke(query.question)

    # 検索結果を整形
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. プロンプトの準備
    prompt_template = """
    以下の情報のみを参考にして、ユーザーの質問に日本語で回答してください。
    情報に答えがない場合は、「その情報は見つかりませんでした」と回答してください。

    --- 参考情報 ---
    {context}
    ---

    ユーザーの質問: {question}
    回答:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 3. LLMに質問と参考情報を渡して回答を生成
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.invoke(
            {"context": context, "question": query.question})
        return {"answer": response['text']}
    except Exception as e:
        return {"error": str(e)}
