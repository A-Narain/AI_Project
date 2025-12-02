# AI_Project

# Explainable Hate Speech Detection System

A transparent AI-powered content moderation system that detects hate speech and provides explanations using BERT, LIME, SHAP, and attention visualization.

## Team Members
- **Satavisha Pan** (23BKT0112) - satavisha.pan2023@vitstudent.ac.in
- **Aastha Narain** (23BCE0237) - aasthanilesh.narain2023@vitstudent.ac.in  
- **Kkomal Padiya** (23BKT0060) - kkomal.padiya2023@vitstudent.ac.in

## Project Overview

This project implements an explainable NLP-based hate speech classifier that:
- Identifies hate speech, offensive language, and neutral content
- Provides transparency through three explainability techniques:
  - **LIME** (Local Interpretable Model-agnostic Explanations)
  - **SHAP** (SHapley Additive exPlanations)
  - **Attention Visualization** from BERT model
- Uses BERT-base-uncased fine-tuned on 25k labeled tweets

## Features

✅ Real-time hate speech detection  
✅ Three explainability methods for transparency  
✅ Word-level importance highlighting  
✅ Confidence scores for predictions  
✅ Modern, responsive UI  
✅ RESTful API backend  

## Tech Stack

### Backend
- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Transformers** (HuggingFace) - BERT implementation
- **Flask** - Web framework
- **LIME & SHAP** - Explainability libraries

### Frontend
- **React** - UI framework
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16+ (for frontend development)
- pip
- 4GB+ RAM recommended
- GPU (optional, but recommended for training)

### Backend Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd hate-speech-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Download the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) from Kaggle
- Save as `hate_speech_dataset.csv` in the project root

5. **Train the model (optional)**
```bash
python train_model.py
```
This will:
- Preprocess the data
- Fine-tune BERT on your dataset
- Save the best model
- Display evaluation metrics

6. **Run the Flask backend**
```bash
python app.py
```
The API will be available at `http://localhost:5000`

### Frontend Setup (for local development)

The demo already works in the artifact viewer, but for local development:

1. **Create React app**
```bash
npx create-react-app hate-speech-frontend
cd hate-speech-frontend
```

2. **Install dependencies**
```bash
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

3. **Configure Tailwind** (tailwind.config.js)
```javascript
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

4. **Copy the React component** from the artifact into `src/App.js`

5. **Update API URL** in the component to `http://localhost:5000`

6. **Start development server**
```bash
npm start
```

## API Documentation

### Endpoints

#### `POST /api/predict`
Analyze text for hate speech

**Request Body:**
```json
{
  "text": "Text to analyze",
  "method": "attention"  // or "lime" or "shap"
}
```

**Response:**
```json
{
  "classification": "hate_speech",
  "confidence": 0.92,
  "probabilities": {
    "hate_speech": 0.92,
    "offensive": 0.05,
    "neither": 0.03
  },
  "explanation": "This text was classified as...",
  "keywords": ["word1", "word2"],
  "word_scores": {
    "word1": 0.95,
    "word2": 0.87
  },
  "method": "attention"
}
```

#### `GET /api/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Dataset

**Source:** [Kaggle - Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

**Size:** ~25,000 labeled tweets

**Classes:**
- 0: Hate Speech
- 1: Offensive Language  
- 2: Neither

**Format:** CSV with columns `tweet` and `class`

## Model Architecture

- **Base Model:** BERT-base-uncased (110M parameters)
- **Task:** Multi-class classification (3 classes)
- **Fine-tuning:** 3 epochs on hate speech dataset
- **Max Sequence Length:** 128 tokens
- **Batch Size:** 16

## Explainability Methods

### 1. Attention Visualization
- Uses BERT's attention weights from the last layer
- Shows which words the model focuses on
- Fast and model-intrinsic

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Perturbs input text to understand importance
- Model-agnostic approach
- Provides local explanations

### 3. SHAP (SHapley Additive exPlanations)
- Based on game theory
- Calculates contribution of each word
- More computationally intensive but theoretically grounded

## Evaluation Metrics

- **Macro F1-Score** (primary metric)
- **Precision** per class
- **Recall** per class
- **Confusion Matrix**
- **Human Agreement Score** (for explainability)

## Project Structure

```
hate-speech-detection/
├── app.py                    # Flask backend
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── hate_speech_dataset.csv # Dataset (download separately)
├── best_model.pt           # Saved model weights
├── hate_speech_model/      # Saved model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── frontend/               # React frontend (optional)
    ├── src/
    │   └── App.js
    ├── package.json
    └── tailwind.config.js
```

## Usage Examples

### Example 1: Neutral Text
```
Input: "I love this community! Everyone is so welcoming."
Output: Neither (95% confidence)
```

### Example 2: Offensive Text
```
Input: "This product is terrible and a complete waste of money."
Output: Offensive (78% confidence)
```

### Example 3: Hate Speech
```
Input: "Those people should not be allowed here because of their religion."
Output: Hate Speech (88% confidence)
```

## Performance

Expected performance metrics after training:
- **Macro F1-Score:** ~0.85-0.90
- **Precision (Hate Speech):** ~0.80-0.85
- **Recall (Hate Speech):** ~0.75-0.82

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch size in `train_model.py`
- Use CPU instead: Change device to `cpu`

**2. Model not loading**
- Ensure you've run training first
- Check model file paths

**3. CORS errors**
- Flask-CORS is properly configured
- Check frontend API URL matches backend

**4. Slow predictions**
- Use GPU if available
- Reduce max_length for faster tokenization
- Use attention method (fastest)

## Future Improvements

- [ ] Add more languages support
- [ ] Implement user feedback loop
- [ ] Add model versioning
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Add batch prediction endpoint
- [ ] Implement A/B testing for explainability methods
- [ ] Add confidence calibration
- [ ] Create mobile app

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## References

1. Davidson, T., et al. (2017). "Automated Hate Speech Detection and the Problem of Offensive Language"
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Ribeiro, M., et al. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (LIME)
4. Lundberg, S., et al. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)

## License

This project is for educational purposes as part of VIT coursework.

## Contact

For questions or issues, please contact:
- Satavisha Pan: satavisha.pan2023@vitstudent.ac.in
- Aastha Narain: aasthanilesh.narain2023@vitstudent.ac.in
- Kkomal Padiya: kkomal.padiya2023@vitstudent.ac.in

## Acknowledgments

- VIT University for project guidance
- HuggingFace for BERT implementation
- Kaggle for the dataset
- The open-source community for explainability libraries
