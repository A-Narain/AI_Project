import { useState } from "react";
import axios from "axios";
import "./index.css";

function App() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");

  const [safeProb, setSafeProb] = useState(0);
  const [harmfulProb, setHarmfulProb] = useState(0);

  const [lime, setLime] = useState([]);
  const [shap, setShap] = useState([]);

  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    if (!text.trim()) return;

    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", {
        text,
      });

      setPrediction(res.data.prediction);
      setConfidence(res.data.confidence);

      setSafeProb(res.data.safe_prob);
      setHarmfulProb(res.data.harmful_prob);

      setLime(res.data.lime || []);
      setShap(res.data.shap || []);
    } catch (err) {
      console.error(err);
      alert("Backend error");
    }

    setLoading(false);
  };

  return (
    <div className="page">
      <div className="card">
        <h1 className="title">Explainable AI Moderation</h1>

        <textarea
          className="textarea"
          rows="6"
          placeholder="Enter text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button className="btn" onClick={analyzeText} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {/* RESULT */}
        {prediction && (
          <div className="resultCard">
            <h2
              className={
                prediction === "Harmful" ? "dangerText" : "safeText"
              }
            >
              {prediction}
            </h2>

            <p>Confidence: {(confidence * 100).toFixed(2)}%</p>

            <p>Safe: {(safeProb * 100).toFixed(2)}%</p>
            <p>Harmful: {(harmfulProb * 100).toFixed(2)}%</p>
          </div>
        )}

        {/* LIME */}
        {lime.length > 0 && (
          <div className="resultCard">
            <h3>LIME Explanation</h3>
            {lime.map((item, i) => (
              <p key={i}>
                <b>{item[0]}</b> → {item[1].toFixed(4)}
              </p>
            ))}
          </div>
        )}

        {/* SHAP */}
        {shap.length > 0 && (
          <div className="resultCard">
            <h3>SHAP Explanation</h3>
            {shap.map((item, i) => (
              <p key={i}>
                <b>{item[0]}</b> → {item[1].toFixed(4)}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;