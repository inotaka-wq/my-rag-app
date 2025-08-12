# backend/main.py

import os
import logging
import uuid
import json
import faiss
import numpy as np  # ★★★ numpyをインポート ★★★
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

# --- 初期設定 ---
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")

vector_store = None
documents_db = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- チャンク設定 ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", "。", "、", " ", ""],
)


def chunkify(content: str, source: str, parent_id: str, tags: Optional[List[str]] = None, category: Optional[str] = None):
    """テキストをチャンクに分割して LangChain Document 配列に変換"""
    chunks = text_splitter.split_text(content)
    docs = []
    ids = []
    for i, ch in enumerate(chunks):
        doc_id = f"{parent_id}::chunk-{i}"
        meta = {
            "source": source,
            "parent_id": parent_id,
            "chunk_index": i,
            "tags": tags or [],
            "category": category or ""
        }
        docs.append(Document(page_content=ch, metadata=meta))
        ids.append(doc_id)
    return ids, docs


def save_knowledge_base():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents_db, f, ensure_ascii=False, indent=2)
    logging.info("ナレッジベースをファイルに保存しました。")


def load_knowledge_base():
    global vector_store, documents_db
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
                documents_db = json.load(f)
            logging.info("ファイルからナレッジベースを正常に読み込みました。")
        except Exception as e:
            logging.error(f"ナレッジベースの読み込みに失敗しました: {e}")
            initialize_empty_db(embeddings)
    else:
        logging.warning("保存されたナレッジベースが見つかりません。空のDBを初期化します。")
        initialize_empty_db(embeddings)


def initialize_empty_db(embeddings):
    global vector_store, documents_db
    embedding_size = 768
    index = faiss.IndexFlatL2(embedding_size)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    vector_store = FAISS(embeddings.embed_query, index,
                         docstore, index_to_docstore_id)
    documents_db = {}


@app.on_event("startup")
def startup_event():
    load_knowledge_base()


class KnowledgeItem(BaseModel):
    content: str
    source: str
    tags: Optional[List[str]] = []
    category: Optional[str] = ""


class AskQuery(BaseModel):
    question: str


TAG_PROMPT = PromptTemplate.from_template("""
次の文章から、社内ナレッジ検索用のタグ(3〜7個)と大分類カテゴリ(1つ)を日本語で抽出してください。
- タグは短い名詞句（例: 休暇, 有休, 勤怠, 経費, セキュリティ）
- 出力はJSONのみ。例: {{"tags":["有休","勤怠"],"category":"労務"}}

文章:
{text}
""")


async def extract_tags_with_llm(text: str):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=TAG_PROMPT)
    try:
        res = await chain.ainvoke({"text": text[:8000]})
        data = json.loads(res["text"].strip())
        tags = [t.strip() for t in data.get("tags", []) if isinstance(t, str)]
        category = data.get("category") or ""
        return tags, category
    except Exception as e:
        logging.warning(f"タグ抽出に失敗: {e}")
        return [], ""

EXPAND_PROMPT = PromptTemplate.from_template("""
ユーザー質問を検索しやすい言い換えに3つ拡張してください。
- それぞれ簡潔な1文
- JSON配列で出力（例: ["…","…","…"]）

質問: {question}
""")


async def expand_queries_with_llm(question: str):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=EXPAND_PROMPT)
    try:
        res = await chain.ainvoke({"question": question[:500]})
        arr = json.loads(res["text"].strip())
        return [str(x) for x in arr][:3]
    except Exception as e:
        logging.warning(f"拡張クエリ生成に失敗: {e}")
        return []


class SummarizeResponse(BaseModel):
    summary: str
    tags: list[str] = []
    category: str = ""


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            loader = UnstructuredFileLoader(temp_path)
        else:
            loader = UnstructuredFileLoader(temp_path)

        docs = loader.load()
        full_text = " ".join([doc.page_content for doc in docs])
        if not full_text.strip():
            raise HTTPException(
                status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        prompt = PromptTemplate.from_template(
            "以下の文章を社内ナレッジとして利用しやすいように、重要ポイントを箇条書きで要約してください。\n\n文章:\n{text}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.ainvoke({"text": full_text[:20000]})
        summary = response["text"]

        tags, category = await extract_tags_with_llm(summary)
        return {"summary": summary, "tags": tags, "category": category}

    except Exception as e:
        logging.error(f"要約エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ファイルの処理中にエラーが発生しました: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/knowledge")
async def get_all_knowledge():
    return [{"id": id, **data} for id, data in documents_db.items()]


@app.post("/knowledge")
async def add_knowledge(item: KnowledgeItem):
    parent_id = str(uuid.uuid4())
    ids, docs = chunkify(item.content, item.source,
                         parent_id, item.tags, item.category)
    vector_store.add_documents(docs, ids=ids)

    documents_db[parent_id] = {
        "content": item.content,
        "source": item.source,
        "chunks": len(ids),
        "chunk_ids": ids,
        "tags": item.tags or [],
        "category": item.category or "",
    }
    save_knowledge_base()
    return {"id": parent_id, **item.dict(), "chunks": len(ids)}


@app.put("/knowledge/{item_id}")
async def update_knowledge(item_id: str, item: KnowledgeItem):
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    old_ids = documents_db[item_id].get("chunk_ids", [])
    if old_ids:
        try:
            vector_store.delete(old_ids)
        except Exception as e:
            logging.warning(f"旧チャンク削除時の警告: {e}")

    ids, docs = chunkify(item.content, item.source,
                         item_id, item.tags, item.category)
    vector_store.add_documents(docs, ids=ids)

    documents_db[item_id] = {
        "content": item.content,
        "source": item.source,
        "chunks": len(ids),
        "chunk_ids": ids,
        "tags": item.tags or [],
        "category": item.category or "",
    }
    save_knowledge_base()
    return {"id": item_id, **item.dict(), "chunks": len(ids)}


@app.delete("/knowledge/{item_id}")
async def delete_knowledge(item_id: str):
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    chunk_ids = documents_db[item_id].get("chunk_ids", [])
    if chunk_ids:
        try:
            vector_store.delete(chunk_ids)
        except Exception as e:
            logging.warning(f"チャンク削除時の警告: {e}")

    del documents_db[item_id]
    save_knowledge_base()
    return {"message": "Knowledge deleted successfully"}


@app.post("/ask")
async def ask_question(query: AskQuery):
    if not vector_store or not documents_db:
        raise HTTPException(status_code=503, detail="ナレッジベースが空です。")

    expanded_queries = await expand_queries_with_llm(query.question)
    all_queries = [query.question] + expanded_queries

    tags, category = await extract_tags_with_llm(query.question)
    logging.debug(f"推定タグ: {tags}, カテゴリ: {category}")

    search_results = []
    # asyncio.gather を使って並列検索
    tasks = []
    for q in all_queries:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
        )
        tasks.append(retriever.ainvoke(q))

    results_from_all_queries = await asyncio.gather(*tasks)
    for docs in results_from_all_queries:
        search_results.extend(docs)

    if tags or category:
        filtered_docs = [
            d for d in search_results
            if any(t in d.metadata.get("tags", []) for t in tags)
            or (category and category == d.metadata.get("category"))
        ]
        # フィルタリングで結果が0件になった場合は、元の検索結果をフォールバックとして使用
        if filtered_docs:
            search_results = filtered_docs

    seen_ids = set()
    unique_results = []
    for d in search_results:
        if d.metadata.get("parent_id") not in seen_ids:
            unique_results.append(d)
            seen_ids.add(d.metadata.get("parent_id"))

    # --- 5. ★★★ 再ランク処理をnumpyを使って修正 ★★★
    if unique_results:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        q_vec = np.array(embeddings.embed_query(
            query.question), dtype=np.float32)

        doc_vecs = [embeddings.embed_query(
            d.page_content) for d in unique_results]
        doc_vecs_np = np.array(doc_vecs, dtype=np.float32)

        # L2距離（ユークリッド距離）を計算
        distances = np.linalg.norm(doc_vecs_np - q_vec, axis=1)

        # 距離が近い順にソート
        sorted_indices = np.argsort(distances)
        ranked_results = [unique_results[i] for i in sorted_indices]
    else:
        ranked_results = []

    top_docs = ranked_results[:6]
    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    logging.debug(f"Context for LLM: {context}")

    prompt_template = """
    あなたは社内ルールに詳しい専門家です。
    提供された『参考情報』だけを根拠として、ユーザーの質問に日本語で回答してください。
    あなたの知識や推測では答えないでください。

    **回答構成**：
    - 分かっていること（参考情報に明記されている事実）
    - 不明な点（記載がないもの）
    - 追加で確認すべき事項

    --- 参考情報 ---
    {context}
    ---

    ユーザーの質問: {question}
    回答:
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = await chain.ainvoke({"context": context, "question": query.question})
        return {
            "answer": response["text"],
            "context": context,
            "expanded_queries": expanded_queries,
            "applied_tags": tags,
            "applied_category": category
        }
    except Exception as e:
        logging.error(f"回答生成エラー: {e}")
        raise HTTPException(status_code=500, detail="回答生成中にエラーが発生しました。")
