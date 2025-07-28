// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("考え中...");
    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      setAnswer(data.answer || data.error);
    } catch (error) {
      setAnswer("APIの呼び出しに失敗しました。");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG App with Gemini</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="質問を入力してください"
            style={{ width: "300px", padding: "10px" }}
          />
          <button type="submit" style={{ padding: "10px" }}>
            質問する
          </button>
        </form>
        {answer && (
          <div
            style={{
              marginTop: "20px",
              padding: "20px",
              border: "1px solid #ccc",
              maxWidth: "600px",
            }}
          >
            <p>
              <strong>回答:</strong>
            </p>
            <p>{answer}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
