import React, { useState, useEffect, useCallback } from "react";
import {
  FileUp,
  MessageSquare,
  Database,
  Trash2,
  Edit,
  Save,
  PlusCircle,
} from "lucide-react";
import "./App.css";

const API_BASE_URL = "http://localhost:8000";

// --- UI Components ---
const TabButton = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 text-lg font-semibold rounded-t-lg transition-colors duration-200 flex items-center gap-2 ${
      active
        ? "bg-white text-blue-600 border-b-2 border-blue-600"
        : "bg-gray-100 text-gray-500 hover:bg-gray-200"
    }`}
  >
    {children}
  </button>
);

const IconButton = ({ onClick, children, className = "" }) => (
  <button
    onClick={onClick}
    className={`p-2 text-gray-500 hover:text-blue-600 transition-colors duration-200 ${className}`}
  >
    {children}
  </button>
);

const Spinner = () => (
  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
);

// --- Knowledge Management Page ---
const KnowledgeManager = () => {
  const [knowledgeList, setKnowledgeList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchKnowledge = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/knowledge`);
      if (!response.ok) throw new Error("Failed to fetch knowledge base.");
      const data = await response.json();
      setKnowledgeList(data);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    fetchKnowledge();
  }, [fetchKnowledge]);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setSummary("");
    setError("");
  };

  const handleSummarize = async () => {
    if (!selectedFile) {
      setError("要約するファイルを選択してください。");
      return;
    }
    setIsLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to summarize file.");
      }
      const data = await response.json();
      setSummary(data.summary);
      setEditingId(null); // 新規要約モード
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!summary.trim()) {
      setError("内容が空です。");
      return;
    }
    setIsLoading(true);
    setError("");
    const endpoint = editingId
      ? `${API_BASE_URL}/knowledge/${editingId}`
      : `${API_BASE_URL}/knowledge`;
    const method = editingId ? "PUT" : "POST";

    try {
      const response = await fetch(endpoint, {
        method: method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: summary,
          source: selectedFile?.name || "Manual Entry",
        }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to save knowledge.");
      }
      await fetchKnowledge();
      handleAddNew();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("本当にこの知識を削除しますか？")) return;
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/knowledge/${id}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("Failed to delete knowledge.");
      await fetchKnowledge();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEdit = (item) => {
    setEditingId(item.id);
    setSummary(item.content);
    setSelectedFile(null);
    setError("");
  };

  const handleAddNew = () => {
    setEditingId(null);
    setSummary("");
    setSelectedFile(null);
    setError("");
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 p-8">
      {/* Left: Editor */}
      <div className="flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-gray-700">ナレッジエディタ</h2>
        <div className="bg-white p-4 rounded-lg shadow-md flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <FileUp className="text-blue-500" />
            <span className="font-semibold">
              Step 1: ファイルから要約 (任意)
            </span>
          </div>
          <input
            type="file"
            onChange={handleFileChange}
            className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          <button
            onClick={handleSummarize}
            disabled={isLoading}
            className="bg-blue-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors duration-200 disabled:bg-blue-300 flex items-center justify-center gap-2"
          >
            {isLoading && !editingId ? <Spinner /> : "AIで要約する"}
          </button>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Edit className="text-green-500" />
            <span className="font-semibold">Step 2: 内容の編集・追記</span>
          </div>
          <textarea
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
            placeholder="ここにAIの要約結果が表示されます。または、直接内容を記述・編集してください。"
            className="w-full h-64 p-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          />
          <button
            onClick={handleSave}
            disabled={isLoading}
            className="bg-green-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-green-600 transition-colors duration-200 disabled:bg-green-300 flex items-center justify-center gap-2"
          >
            {isLoading ? <Spinner /> : <Save />}
            {editingId ? "更新する" : "新規登録する"}
          </button>
          {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
      </div>

      {/* Right: Knowledge List */}
      <div className="flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-700">
            登録済みナレッジ一覧
          </h2>
          <button
            onClick={handleAddNew}
            className="bg-gray-200 text-gray-700 font-bold py-2 px-4 rounded-lg hover:bg-gray-300 flex items-center gap-2"
          >
            <PlusCircle size={20} /> 新規作成
          </button>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md h-[calc(100vh-12rem)] overflow-y-auto">
          {knowledgeList.length === 0 ? (
            <p className="text-gray-500">登録済みのナレッジはありません。</p>
          ) : (
            <ul className="space-y-3">
              {knowledgeList.map((item) => (
                <li
                  key={item.id}
                  className="p-3 bg-gray-50 rounded-lg border flex justify-between items-start gap-2"
                >
                  <div className="flex-grow">
                    <p className="text-sm text-gray-400">ID: {item.id}</p>
                    <p className="text-gray-800 whitespace-pre-wrap">
                      {item.content}
                    </p>
                  </div>
                  <div className="flex-shrink-0 flex">
                    <IconButton onClick={() => handleEdit(item)}>
                      <Edit size={18} />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(item.id)}>
                      <Trash2
                        size={18}
                        className="text-red-400 hover:text-red-600"
                      />
                    </IconButton>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

// --- Chat Page ---
const Chat = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    setAnswer("");
    setContext("");
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to get answer.");
      }
      const data = await response.json();
      setAnswer(data.answer);
      setContext(data.context);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-8 flex flex-col items-center">
      <div className="w-full max-w-3xl">
        <h1 className="text-3xl font-bold text-center text-gray-700 mb-6">
          AI社内ルールQA
        </h1>
        <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="質問を入力してください (例: 始末書の提出ルールは？)"
            className="flex-grow p-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-blue-500 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-600 transition-colors duration-200 disabled:bg-blue-300 flex items-center justify-center"
          >
            {isLoading ? <Spinner /> : "質問する"}
          </button>
        </form>

        {error && (
          <div
            className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-4"
            role="alert"
          >
            {error}
          </div>
        )}

        {answer && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold text-gray-800 mb-2">回答</h2>
            <p className="text-gray-700 whitespace-pre-wrap">{answer}</p>

            {context && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-800">
                  参照したナレッジを表示
                </summary>
                <div className="mt-2 p-3 bg-gray-50 border rounded-md text-xs text-gray-600 whitespace-pre-wrap">
                  {context}
                </div>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// --- Main App Component ---
function App() {
  const [activeTab, setActiveTab] = useState("chat");

  return (
    <div className="bg-gray-100 min-h-screen font-sans">
      <header className="bg-white shadow-md">
        <nav className="container mx-auto px-6 py-3 flex">
          <TabButton
            active={activeTab === "chat"}
            onClick={() => setActiveTab("chat")}
          >
            <MessageSquare size={20} />
            <span>AIチャット</span>
          </TabButton>
          <TabButton
            active={activeTab === "admin"}
            onClick={() => setActiveTab("admin")}
          >
            <Database size={20} />
            <span>ナレッジ管理</span>
          </TabButton>
        </nav>
      </header>
      <main>
        {activeTab === "chat" && <Chat />}
        {activeTab === "admin" && <KnowledgeManager />}
      </main>
    </div>
  );
}

export default App;
