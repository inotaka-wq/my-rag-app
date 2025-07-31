# backend/main.py

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChainとFAISS関連のライブラリをインポート
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# ★★★ ハイブリッド検索のためのライブラリをインポート ★★★
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ロガーの基本設定
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 環境変数の読み込みとAPIキーの設定
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- グローバル変数 ---
# ★★★ retrieverをグローバル変数として保持 ★★★
retriever = None

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
    global retriever  # グローバル変数を参照
    logging.info("サーバー起動時にRAGのセットアップを開始します...")

    # 1. ドキュメントの読み込み
    # (省略...前回と同じ)
    pdf_loader = DirectoryLoader(
        './docs', glob="**/*.pdf", loader_cls=UnstructuredFileLoader, show_progress=True)
    excel_loader = DirectoryLoader(
        './docs', glob="**/*.xlsx", loader_cls=UnstructuredFileLoader, show_progress=True)
    documents = pdf_loader.load()
    documents.extend(excel_loader.load())

    if not documents:
        logging.warning("ドキュメントが見つかりませんでした。")
        return
    else:
        loaded_files = set(doc.metadata.get('source', '不明なファイル')
                           for doc in documents)
        logging.info(f"--- 読み込み完了したファイル ({len(loaded_files)}件) ---")
        for file_path in loaded_files:
            logging.info(f"- {os.path.basename(file_path)}")
        logging.info("------------------------------------")

    # 2. ドキュメントの分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logging.info(f"ドキュメントを {len(texts)}個のチャンクに分割しました。")

    # 3. ベクトル化（Embedding）モデルの準備
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. ★★★ ハイブリッド検索リトリーバーの作成 ★★★
    # 4-1. キーワード検索 (BM25) のリトリーバーを作成
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 5  # キーワード検索で上位5件を取得

    # 4-2. ベクトル検索 (FAISS) のリトリーバーを作成
    vector_store = FAISS.from_documents(texts, embeddings)
    faiss_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5})  # ベクトル検索で上位5件を取得

    # 4-3. 2つのリトリーバーを組み合わせる (EnsembleRetriever)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # キーワード検索とベクトル検索を50%:50%の重みで評価
    )

    logging.info("ハイブリッド検索リトリーバーのセットアップが完了しました。")


# --- APIエンドポイント ---
class Query(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "RAG API is running."}


@app.post("/debug-retrieval")
async def debug_retrieval(query: Query):
    global retriever  # グローバル変数を参照
    if retriever is None:
        return {"error": "リトリーバーが準備できていません。"}

    retrieved_docs = retriever.invoke(query.question)

    results = []
    for doc in retrieved_docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })

    return {"retrieved_chunks": results}


@app.post("/ask")
async def ask(query: Query):
    global retriever  # グローバル変数を参照
    if retriever is None:
        return {"error": "リトリーバーが準備できていません。"}

    # 1. ハイブリッド検索を実行
    retrieved_docs = retriever.invoke(query.question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. プロンプトの準備 (変更なし)
    prompt_template = """
    あなたは人事労務の専門家です。提供された『参考情報』に書かれている内容だけを根拠として、ユーザーの質問に回答してください。
    あなたの知識や推測で回答してはいけません。

    『参考情報』に記載がない、または質問に回答するための情報が不足している場合は、必ず「就業規則には該当する記載がありませんでした」とだけ回答してください。

    --- 参考情報 ---
    {context}
    ---

    ユーザーの質問: {question}
    回答:
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # 3. LLMに質問と参考情報を渡して回答を生成 (変更なし)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.invoke(
            {"context": context, "question": query.question})
        return {"answer": response['text'], "context": context}
    except Exception as e:
        return {"error": str(e), "context": context}
