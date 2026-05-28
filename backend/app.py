from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

from lime.lime_text import LimeTextExplainer

# =========================
# APP SETUP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL LOAD
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
model = DistilBertForSequenceClassification.from_pretrained("./model")
model.eval()

# =========================
# LIME SETUP
# =========================
class_names = ["Safe", "Harmful"]
lime_explainer = LimeTextExplainer(class_names=class_names)

# =========================
# REQUEST MODEL
# =========================
class TextRequest(BaseModel):
    text: str


# =========================
# PREDICTION FUNCTION
# =========================
def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.numpy()


# =========================
# LIME EXPLANATION
# =========================
def get_lime_explanation(text):
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=8
    )
    return exp.as_list()


# =========================
# SHAP (SAFE LIGHTWEIGHT VERSION)
# Token contribution approximation
# =========================
def get_shap_explanation(text):
    tokens = text.split()

    base_prob = predict_proba([text])[0]
    base_score = float(base_prob[1])  # Harmful probability

    contributions = []

    # MASK-STYLE APPROXIMATION (fast + stable)
    for i, token in enumerate(tokens):
        masked_text = " ".join([t for j, t in enumerate(tokens) if j != i])

        if len(masked_text.strip()) == 0:
            score = base_score
        else:
            prob = predict_proba([masked_text])[0][1]
            score = base_score - float(prob)

        contributions.append((token, round(score, 4)))

    # sort by impact
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return contributions[:8]


# =========================
# ENDPOINT
# =========================
@app.post("/predict")
def predict(request: TextRequest):

    text = request.text

    probs = predict_proba([text])[0]

    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = "Harmful" if pred_class == 1 else "Safe"

    # LIME
    lime_exp = get_lime_explanation(text)

    # SHAP (approx)
    shap_exp = get_shap_explanation(text)

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "safe_prob": round(float(probs[0]), 4),
        "harmful_prob": round(float(probs[1]), 4),

        "lime": lime_exp,
        "shap": shap_exp
    }