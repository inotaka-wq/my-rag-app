# backend/main.py

import os
import logging
import uuid
import json
import faiss
import numpy as np
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
# アプリケーション全体のログ設定。DEBUGレベル以上のログを全てコンソールに出力する。
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")
# .envファイルから環境変数を読み込む
load_dotenv()
# 環境変数からGoogle APIキーを読み込む。なければ空文字を設定。
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# --- グローバル変数 & データ永続化設定 ---
# 永続化データを保存するディレクトリ名
DATA_DIR = "data"
# FAISS（ベクトルDB）のインデックスを保存するファイルパス
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
# 人間が管理するナレッジ本文（親ドキュメント）を保存するJSONファイルパス
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")

# アプリケーション全体で共有するベクトルストアのインスタンス
vector_store = None
# 親ドキュメント情報を管理するための辞書（インメモリDB）
# このDBは、UIでの一覧表示や、更新・削除時のチャンク特定に使用される。
documents_db = {}

# --- FastAPIアプリの初期化 ---
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) の設定。
# localhost:3000 (React開発サーバー) からのAPIアクセスを許可する。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- チャンク設定 ---
# ナレッジを分割する際の基本設定
CHUNK_SIZE = 800  # 1チャンクあたりの最大文字数
CHUNK_OVERLAP = 100  # チャンク間の重複文字数。文脈が途切れるのを防ぐ。

# LangChainのテキスト分割ツールを初期化
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # 分割を試みる区切り文字の優先順位。改行や句読点を優先する。
    separators=["\n\n", "\n", "。", "、", " ", ""],
)


def chunkify(content: str, source: str, parent_id: str, tags: Optional[List[str]] = None, category: Optional[str] = None):
    """
    1つのナレッジ本文（親）を、検索用の小さなチャンク（子）に分割する関数。
    Args:
        content (str): ナレッジの全文。
        source (str): 元のファイル名など。
        parent_id (str): このチャンク群が属する親ナレッジのID。
        tags (Optional[List[str]]): このナレッジに付与されたタグ。
        category (Optional[str]): このナレッジのカテゴリ。
    Returns:
        Tuple[List[str], List[Document]]: (チャンクIDのリスト, LangChain Documentオブジェクトのリスト)
    """
    # テキストをチャンクに分割
    chunks = text_splitter.split_text(content)
    docs = []
    ids = []
    # 各チャンクに対して、IDとメタデータ（戸籍情報）を付与する
    for i, ch in enumerate(chunks):
        doc_id = f"{parent_id}::chunk-{i}"  # 親IDと連番でユニークなIDを作成
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

# --- データ永続化ヘルパー関数 ---


def save_knowledge_base():
    """FAISSインデックスとドキュメントDBをファイルに保存する。"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents_db, f, ensure_ascii=False, indent=2)
    logging.info("ナレッジベースをファイルに保存しました。")


def load_knowledge_base():
    """ファイルからFAISSインデックスとドキュメントDBを読み込む。"""
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
    """空のFAISSインデックスとドキュメントDBを新規作成する。"""
    global vector_store, documents_db
    embedding_size = 768  # GoogleのEmbeddingモデルのベクトル次元数
    index = faiss.IndexFlatL2(embedding_size)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    vector_store = FAISS(embeddings.embed_query, index,
                         docstore, index_to_docstore_id)
    documents_db = {}

# --- サーバー起動時の処理 ---


@app.on_event("startup")
def startup_event():
    """FastAPIサーバー起動時に一度だけ実行される処理。"""
    load_knowledge_base()

# --- APIリクエスト/レスポンスモデル定義 ---


class KnowledgeItem(BaseModel):
    """ナレッジ登録・更新時にフロントエンドから受け取るデータモデル。"""
    content: str
    source: str
    tags: Optional[List[str]] = []
    category: Optional[str] = ""


class AskQuery(BaseModel):
    """質問時にフロントエンドから受け取るデータモデル。"""
    question: str

# --- LLMを使った補助機能 ---


# AIにタグとカテゴリを抽出させるための指示書（プロンプト）
TAG_PROMPT = PromptTemplate.from_template("""
次の文章から、社内ナレッジ検索用のタグ(3〜7個)と大分類カテゴリ(1つ)を日本語で抽出してください。
- タグは短い名詞句（例: 休暇, 有休, 勤怠, 経費, セキュリティ）
- 出力はJSONのみ。例: {{"tags":["有休","勤怠"],"category":"労務"}}

文章:
{text}
""")


async def extract_tags_with_llm(text: str):
    """LLMを使ってテキストからタグとカテゴリを抽出する非同期関数。"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=TAG_PROMPT)
    try:
        # テキストが長すぎるとエラーになるため制限
        res = await chain.ainvoke({"text": text[:8000]})
        data = json.loads(res["text"].strip())
        tags = [t.strip() for t in data.get("tags", []) if isinstance(t, str)]
        category = data.get("category") or ""
        return tags, category
    except Exception as e:
        logging.warning(f"タグ抽出に失敗: {e}")
        return [], ""

# AIにユーザーの質問を言い換えさせるための指示書（プロンプト）
EXPAND_PROMPT = PromptTemplate.from_template("""
ユーザー質問を検索しやすい言い換えに3つ拡張してください。
- それぞれ簡潔な1文
- JSON配列で出力（例: ["…","…","…"]）

質問: {question}
""")


async def expand_queries_with_llm(question: str):
    """LLMを使ってユーザーの質問を複数の異なる聞き方に拡張する非同期関数。"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = LLMChain(llm=llm, prompt=EXPAND_PROMPT)
    try:
        res = await chain.ainvoke({"question": question[:500]})
        arr = json.loads(res["text"].strip())
        return [str(x) for x in arr][:3]
    except Exception as e:
        logging.warning(f"拡張クエリ生成に失敗: {e}")
        return []

# --- APIエンドポイント定義 ---


class SummarizeResponse(BaseModel):
    """/summarize APIのレスポンスモデル。"""
    summary: str
    tags: list[str] = []
    category: str = ""


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_file(file: UploadFile = File(...)):
    """
    ファイルアップロードを受け取り、内容を要約し、タグとカテゴリを付けて返すAPI。
    """
    temp_path = f"temp_{file.filename}"
    try:
        # アップロードされたファイルを一時的に保存
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # ファイル形式に応じて適切なローダーを選択
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:  # Excel, Txt, Wordなど、多くの形式に対応
            loader = UnstructuredFileLoader(temp_path)

        docs = loader.load()
        full_text = " ".join([doc.page_content for doc in docs])
        if not full_text.strip():
            raise HTTPException(
                status_code=400, detail="ファイルからテキストを抽出できませんでした。")

        # Geminiに要約を依頼
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        prompt = PromptTemplate.from_template(
            "以下の文章を社内ナレッジとして利用しやすいように、重要ポイントを箇条書きで要約してください。\n\n文章:\n{text}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.ainvoke({"text": full_text[:20000]})  # 長文対策
        summary = response["text"]

        # 生成した要約から、さらにタグとカテゴリを抽出
        tags, category = await extract_tags_with_llm(summary)
        return {"summary": summary, "tags": tags, "category": category}

    except Exception as e:
        logging.error(f"要約エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ファイルの処理中にエラーが発生しました: {str(e)}")
    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/knowledge")
async def get_all_knowledge():
    """登録済みの全ナレッジ（親ドキュメント）の一覧を返すAPI。"""
    return [{"id": id, **data} for id, data in documents_db.items()]


@app.post("/knowledge")
async def add_knowledge(item: KnowledgeItem):
    """新しいナレッジをDBに登録するAPI。"""
    parent_id = str(uuid.uuid4())

    # 1. ナレッジをチャンクに分割
    ids, docs = chunkify(item.content, item.source,
                         parent_id, item.tags, item.category)

    # 2. チャンクをベクトルDBに追加
    vector_store.add_documents(docs, ids=ids)

    # 3. 親ドキュメントの情報を管理DBに追加
    documents_db[parent_id] = {
        "content": item.content,
        "source": item.source,
        "chunks": len(ids),
        "chunk_ids": ids,  # どのチャンクに分割されたかを記録
        "tags": item.tags or [],
        "category": item.category or "",
    }
    save_knowledge_base()
    return {"id": parent_id, **item.dict(), "chunks": len(ids)}


@app.put("/knowledge/{item_id}")
async def update_knowledge(item_id: str, item: KnowledgeItem):
    """既存のナレッジを更新するAPI。"""
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    # 1. 古いチャンクをベクトルDBから正確に削除
    old_ids = documents_db[item_id].get("chunk_ids", [])
    if old_ids:
        try:
            vector_store.delete(old_ids)
        except Exception as e:
            logging.warning(f"旧チャンク削除時の警告: {e}")

    # 2. 新しい内容で再チャンク化して登録
    ids, docs = chunkify(item.content, item.source,
                         item_id, item.tags, item.category)
    vector_store.add_documents(docs, ids=ids)

    # 3. 管理DBの親ドキュメント情報を更新
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
    """ナレッジを（親と関連する全チャンクを含めて）削除するAPI。"""
    if item_id not in documents_db:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    # 1. 削除対象の全チャンクIDを取得
    chunk_ids = documents_db[item_id].get("chunk_ids", [])
    if chunk_ids:
        try:
            vector_store.delete(chunk_ids)
        except Exception as e:
            logging.warning(f"チャンク削除時の警告: {e}")

    # 2. 管理DBから親ドキュメント情報を削除
    del documents_db[item_id]
    save_knowledge_base()
    return {"message": "Knowledge deleted successfully"}


@app.post("/ask")
async def ask_question(query: AskQuery):
    """質問を受け取り、多段検索を経て回答を生成するメインAPI。"""
    if not vector_store or not documents_db:
        raise HTTPException(status_code=503, detail="ナレッジベースが空です。")

    # --- ステップ1: クエリ拡張 ---
    # ユーザーの質問を、AIを使って複数の異なる聞き方に変換する。
    expanded_queries = await expand_queries_with_llm(query.question)
    all_queries = [query.question] + expanded_queries
    logging.debug(f"拡張クエリ: {expanded_queries}")

    # --- ステップ2: タグ・カテゴリ推定 ---
    # 質問内容から、関連しそうなタグやカテゴリをAIに推測させる。
    tags, category = await extract_tags_with_llm(query.question)
    logging.debug(f"推定タグ: {tags}, カテゴリ: {category}")

    # --- ステップ3: 複数クエリで並列検索 ---
    # 元の質問と拡張した質問、全てを使って同時に検索をかけ、候補を広く集める。
    search_results = []
    tasks = []
    for q in all_queries:
        retriever = vector_store.as_retriever(
            search_type="mmr",  # MMR: 関連性が高く、かつ多様な結果を返す検索アルゴリズム
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
        )
        tasks.append(retriever.ainvoke(q))

    # asyncio.gatherで、複数の検索処理を並列実行し、高速化する。
    results_from_all_queries = await asyncio.gather(*tasks)
    for docs in results_from_all_queries:
        search_results.extend(docs)

    # --- ステップ4: タグ・カテゴリによるフィルタリング ---
    # AIが推定したタグやカテゴリを使って、検索結果をさらに絞り込む。
    if tags or category:
        filtered_docs = [
            d for d in search_results
            if any(t in d.metadata.get("tags", []) for t in tags)
            or (category and category == d.metadata.get("category"))
        ]
        # フィルタリングで結果が0件になった場合は、元の検索結果をフォールバックとして使用する。
        if filtered_docs:
            search_results = filtered_docs

    # --- ステップ5: 重複排除 ---
    # 異なるチャンクでも、同じ親ナレッジに属する場合は1つにまとめる。
    seen_parent_ids = set()
    unique_results = []
    for d in search_results:
        parent_id = d.metadata.get("parent_id")
        if parent_id not in seen_parent_ids:
            unique_results.append(d)
            seen_parent_ids.add(parent_id)

    # --- ステップ6: 再ランキング (Re-ranking) ---
    # 絞り込んだ候補の中から、改めて「元の質問」と意味的に最も近いものを厳選し、順位を付け直す。
    if unique_results:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        q_vec = np.array(embeddings.embed_query(
            query.question), dtype=np.float32).reshape(1, -1)

        doc_vecs = np.array([embeddings.embed_query(d.page_content)
                            for d in unique_results], dtype=np.float32)

        # L2距離（ユークリッド距離）を計算して、ベクトル間の近さを測る。
        distances = np.linalg.norm(doc_vecs - q_vec, axis=1)

        # 距離が近い順（昇順）にソートする。
        sorted_indices = np.argsort(distances)
        ranked_results = [unique_results[i] for i in sorted_indices]
    else:
        ranked_results = []

    # --- ステップ7: コンテキスト化と回答生成 ---
    top_docs = ranked_results[:6]  # 最終的に上位6件をコンテキストとして使用
    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    logging.debug(f"最終コンテキスト: {context}")

    # LLMへの最終的な指示書。構造化された回答を要求している点が秀逸。
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
        # フロントエンドに返す情報。デバッグ用に拡張クエリや適用タグも返す親切設計。
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
