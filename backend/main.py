# backend/main.py

import os
import logging
import uuid
import json
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- グローバル変数 & データ永続化設定 ---
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")

vector_store = None
documents_db = {}  # { "uuid": {"content": "...", "source": "..."} }

# --- FastAPIアプリの初期化 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ヘルパー関数 ---


def save_knowledge_base():
    """FAISSインデックスとドキュメントDBをファイルに保存"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(documents_db, f, ensure_ascii=False, indent=2)
    logging.info("ナレッジベースをファイルに保存しました。")


def load_knowledge_base():
    """ファイルからFAISSインデックスとドキュメントDBを読み込む"""
    global vector_store, documents_db
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
                documents_db = json.load(f)
            logging.info("ファイルからナレッジベースを正常に読み込みました。")
        except Exception as e:
            logging.error(f"ナレッジベースの読み込みに失敗しました: {e}")
            initialize_empty_db(embeddings)
    else:
        logging.warning("保存されたナレッジベースが見つかりません。空のDBを初期化します。")
        initialize_empty_db(embeddings)


def initialize_empty_db(embeddings):
    """空のFAISSインデックスとドキュメントDBを作成"""
    global vector_store, documents_db
    embedding_size = 768  # GoogleGenerativeAIEmbeddingsのデフォルト次元数
    index = faiss.IndexFlatL2(embedding_size)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    vector_store = FAISS(embeddings.embed_query, index,
                         docstore, index_to_docstore_id)
    documents_db = {}


# --- サーバー起動時の処理 ---
@app.on_event("startup")
def startup_event():
    load_knowledge_base()

# --- APIモデル定義 ---


class KnowledgeItem(BaseModel):
    content: str
    source: str


class AskQuery(BaseModel):
    question: str

# --- APIエンドポイント ---


@app.post("/summarize")
async def summarize_file(file: UploadFile = File(...)):
    """アップロードされたファイルを読み込み、内容を要約する"""
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        if file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            # UnstructuredFileLoaderがExcelをうまく扱える
            loader = UnstructuredFileLoader(temp_path)
        else:
            loader = UnstructuredFileLoader(temp_path)

        docs = loader.load()
        full_text = " ".join([doc.page_content for doc in docs])

        if not full_text.strip():
            raise HTTPException(
                status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        # Geminiに要約を依頼
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        prompt = PromptTemplate.from_template(
            "以下の文章を、社内ナレッジとして利用しやすいように、重要なポイントを箇条書きで分かりやすく要約してください。\n\n文章:\n{text}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.ainvoke({"text": full_text[:20000]})  # 長文対策

        return {"summary": response['text']}

    except Exception as e:
        logging.error(f"要約エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ファイルの処理中にエラーが発生しました: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/knowledge")
async def get_all_knowledge():
    """登録済みの全ナレッジを取得"""
    return [{"id": id, **data} for id, data in documents_db.items()]


@app.post("/knowledge")
async def add_knowledge(item: KnowledgeItem):
    """新しいナレッジをDBに登録"""
    doc_id = str(uuid.uuid4())
    doc = Document(page_content=item.content, metadata={"source": item.source})
    vector_store.add_documents([doc], ids=[doc_id])
    documents_db[doc_id] = {"content": item.content, "source": item.source}
    save_knowledge_base()
    return {"id": doc_id, **item.dict()}


@app.put("/knowledge/{item_id}")
async def update_knowledge(item_id: str, item: KnowledgeItem):
    """既存のナレッジを更新"""
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    # FAISSでは直接更新が難しいため、一度削除して再追加する
    vector_store.delete([item_id])
    doc = Document(page_content=item.content, metadata={"source": item.source})
    vector_store.add_documents([doc], ids=[item_id])

    documents_db[item_id] = {"content": item.content, "source": item.source}
    save_knowledge_base()
    return {"id": item_id, **item.dict()}


@app.delete("/knowledge/{item_id}")
async def delete_knowledge(item_id: str):
    """ナレッジを削除"""
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    vector_store.delete([item_id])
    del documents_db[item_id]
    save_knowledge_base()
    return {"message": "Knowledge deleted successfully"}


@app.post("/ask")
async def ask_question(query: AskQuery):
    """質問を受け取り、ナレッジベースを検索して回答を生成"""
    if not vector_store or not documents_db:
        raise HTTPException(
            status_code=503, detail="ナレッジベースが空です。管理画面からナレッジを登録してください。")

    # 1. 質問に関連するナレッジをベクトル検索
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query.question)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. プロンプトを作成
    prompt_template = """
    あなたは社内ルールに詳しい専門家です。提供された『参考情報』だけを根拠として、ユーザーの質問に日本語で回答してください。
    あなたの知識や推測で回答してはいけません。

    回答は、ユーザーが読みやすいように、マークダウン形式を使用し、箇条書きや改行を適切に使って整形してください。

    『参考情報』に記載がない、または質問に回答するための情報が不足している場合は、必ず「ナレッジベースに該当する情報が見つかりませんでした」と回答してください。

    --- 参考情報 ---
    {context}
    ---

    ユーザーの質問: {question}
    回答:
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # 3. LLMで回答を生成
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = await chain.ainvoke({"context": context, "question": query.question})
        return {"answer": response['text'], "context": context}
    except Exception as e:
        logging.error(f"回答生成エラー: {e}")
        raise HTTPException(status_code=500, detail="回答の生成中にエラーが発生しました。")
