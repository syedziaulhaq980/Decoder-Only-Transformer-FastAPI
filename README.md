# Decoder-Only Transformer + FastAPI

A beginner-friendly implementation of a **GPT-style Decoder-Only Transformer** built from scratch using TensorFlow and exposed through a **FastAPI REST API**.

The project demonstrates the complete flow from text input to next-token prediction and API deployment.

## 🚀 Live Demo

**Live API:**
https://decoder-only-transformer-fastapi.onrender.com/

**Interactive API Documentation:**
https://decoder-only-transformer-fastapi.onrender.com/docs

## 📌 Project Overview

The goal of this project was to understand the core architecture of a Decoder-Only Transformer and learn how to connect a trained deep learning model with a REST API.

The Transformer was implemented using TensorFlow and includes:

* Token Embeddings
* Positional Embeddings
* Multi-Head Self-Attention
* Causal Masking
* Padding Masking
* Feed Forward Network
* Residual Connections
* Layer Normalization
* Stacked Decoder Blocks
* Next-Token Prediction

The trained model was then integrated with FastAPI and deployed on Render.

## 🧠 Transformer Architecture

```text
Input Text
    ↓
Text Vectorization
    ↓
Token IDs
    ↓
Token Embedding
    +
Positional Embedding
    ↓
Causal Multi-Head Self-Attention
    ↓
Add & Normalize
    ↓
Feed Forward Network
    ↓
Add & Normalize
    ↓
Decoder Block 1
    ↓
Decoder Block 2
    ↓
Final Layer Normalization
    ↓
Vocabulary Projection
    ↓
Next-Token Prediction
```

## ⚙️ Model Configuration

| Parameter               |  Value |
| ----------------------- | -----: |
| Vocabulary Size         | 20,000 |
| Maximum Sequence Length |     64 |
| Embedding Dimension     |    128 |
| Attention Heads         |      4 |
| Decoder Blocks          |      2 |
| Feed Forward Dimension  |    256 |
| Dropout                 |    0.1 |

## 📊 Training

The model was trained using a **next-token prediction** objective with sparse categorical cross-entropy loss.

Example:

```text
Input:
i feel very

Target:
feel very happy
```

Training test result:

```text
Training Loss:    2.6612
Validation Loss:  2.3602
```

The model weights were saved and later loaded for API inference.

## 🌐 FastAPI Integration

The trained Transformer is exposed through a simple REST API.

### Endpoint

```text
POST /predict
```

### Request

```json
{
  "text": "i feel"
}
```

### Example Response

```json
{
  "input_text": "i feel",
  "next_token": "i"
}
```

The predicted token depends on the trained model.

## 🔄 API Workflow

```text
Client
  ↓
POST /predict
  ↓
FastAPI
  ↓
Text Input Validation
  ↓
Text Vectorization
  ↓
Decoder-Only Transformer
  ↓
Saved Model Weights
  ↓
Next-Token Prediction
  ↓
JSON Response
```

## 📂 Project Structure

```text
Decoder-Only-Transformer-FastAPI/
│
├── decoder_only_transformer.ipynb
├── transformer_model.py
├── main.py
├── decoder_transformer.weights.h5
├── vectorizer_vocab.txt
├── requirements.txt
├── .python-version
├── .gitignore
└── venv/
```

> `venv/` is excluded from Git using `.gitignore`.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* FastAPI
* Uvicorn
* Pydantic
* NLP
* Transformer Architecture
* Render

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/syedziaulhaq980/Decoder-Only-Transformer-FastAPI.git
```

### 2. Move into the project

```bash
cd Decoder-Only-Transformer-FastAPI
```

### 3. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the API

```bash
uvicorn main:app --reload
```

### 6. Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

## 📚 What I Learned

Through this project, I practiced:

* Understanding Decoder-Only Transformer architecture
* Token and positional embeddings
* Causal self-attention
* Preventing future-token information leakage
* Multi-head attention
* Feed forward networks
* Residual connections and LayerNorm
* Next-token prediction
* Training and validation
* Saving and loading model weights
* Building REST APIs with FastAPI
* Connecting a trained ML model to an API
* Deploying an ML API using Render

## ⚠️ Limitations

This is a **small educational Transformer implementation**, not a full-scale GPT model.

The model was trained for a limited number of steps using an emotion-text dataset, so its next-token predictions are limited and are not intended to match the capabilities of large language models.

The main purpose of this project is to demonstrate understanding of the **Decoder-Only Transformer architecture, model inference, FastAPI integration, and deployment**.

## 🔮 Future Improvements

* Train on a dedicated language-modeling dataset
* Train for more steps
* Improve next-token prediction quality
* Add multi-token text generation
* Add temperature or sampling-based generation
* Create a simple frontend
* Improve deployment and model hosting

## 👨‍💻 Author

**Syed Ziaul Haq**

GitHub:
https://github.com/syedziaulhaq980
