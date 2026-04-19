<div align="center">
  <h1>🧠 Digital Brain</h1>
  <p><b>A Personalized Multimodal AI Memory and Recognition System</b></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/Flask-API-lightgrey?style=for-the-badge&logo=flask" alt="Flask">
    <img src="https://img.shields.io/badge/React-19-cyan?style=for-the-badge&logo=react" alt="React">
    <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge" alt="ChromaDB">
    <img src="https://img.shields.io/badge/Llama_3.1-LLM-purple?style=for-the-badge&logo=meta" alt="Llama 3.1">
  </p>
</div>

---

## 📖 Overview

**Digital Brain** is an advanced, production-grade multimodal personal AI system. It utilizes cutting-edge computer vision, speech recognition, and Retrieval-Augmented Generation (RAG) to seamlessly identify users and maintain a highly personalized, context-aware memory of all interactions.

Unlike standard chatbots, Digital Brain *remembers who you are*. It recognizes you via webcam or microphone, extracts permanent lifecycle facts through asynchronous data mining, and securely indexes your personal PDF documents and chat notes into an episodic memory vector store.

---

## ✨ Enterprise-Grade Features

* 👁️ **Biometric Identity Layer**: Real-time facial extraction (via DeepFace MTCNN) and voice speaker embeddings (via SpeechBrain ECAPA-TDNN).
* 🧠 **Retrieval-Augmented Generation (RAG)**: Upload documents (PDF/TXT) and user notes. Chunks are embedded locally via `sentence-transformers` avoiding external leaking and stored in ChromaDB for bounded semantic search.
* ⚡ **Asynchronous Perception & Cognition**: Heavy workloads (like audio/video embedding, background fact extraction) run concurrently using `ThreadPoolExecutor` avoiding Flask request blocking.
* 💾 **Dual-Memory Architecture**: 
  * *Relational (SQLite)*: Immediate short-term chat rolling windows, persistent key-value facts, and user auth routing.
  * *Vector (ChromaDB)*: Long-term episodic memory, RAG contexts, and high-dimensional biometric tensor matching.
* 🛡️ **Prompt Injection Security**: RAG structures are safely boundary-tagged to prevent context-based prompt subversion on the main LLM.

---

## 🏗️ System Architecture

```mermaid
graph TD;
    Client[React Frontend] -->|Base64 Media & Chat| API[Flask API Routes]
    API -->|Concurrent Threads| Perception[Perception Engine]
    API --> Cognition[Cognition & RAG Engine]
    
    subgraph Perception Engine
        Face[VisionModule: DeepFace]
        Voice[VoiceModule: SpeechBrain]
    end
    
    subgraph Database Layer
        RDB[(SQLite: Working Memory)]
        VDB[(ChromaDB: Episodic Memory)]
    end
    
    Perception --> VDB
    Cognition <-->|Context Search & Fact Extract| Database Layer
    Cognition -->|Injected Prompt| LLM[NVIDIA / LLaMA 3.1 LLM]
```

---

## 💻 Tech Stack

### Backend
* **Core**: `Python 3.12`, `Flask`, `Flask-CORS`
* **AI / ML**: `DeepFace` (ResNet/MTCNN), `SpeechBrain` (torchaudio), `sentence-transformers`
* **Database**: `ChromaDB` (Local Vector Store), `SQLite` (Relational Working DB)
* **LLM Ops**: `OpenAI Client` connecting to hosted `meta/llama-3.1-8b-instruct`, `LangChain Text Splitters`

### Frontend
* **Core**: `React 19`, `Vite`
* **Media UI**: `react-webcam`
* **Styling**: Vanilla CSS with modern dynamic animations

---

## ⚙️ Setup & Installation

### 1. Repository Setup
```bash
git clone https://github.com/krupakargurije/Digital-Brain.git
cd Digital-Brain
```

### 2. Python Environment Setup
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root of the project to securely provide your API keys.
```env
GOOGLE_API_KEY=your_gemini_key_here
NVIDIA_API_KEY=your_nvidia_llama_key_here
```

### 4. Start the Application
**Terminal 1 (Backend - Headless API)**
```bash
python app.py
```

**Terminal 2 (Frontend - React UI)**
```bash
cd digital-brain-ui
npm install
npm run dev
```

---

## 🔌 Core API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/identify` | Takes Base64 Audio/Video. Extracts embeddings, registers or returns recognized user profile. |
| `POST` | `/api/chat` | Main LLM conversational loop. Initiates async fact extraction and RAG memory retrieval. |
| `POST` | `/api/poll_vision` | Silent endpoint to continuously monitor via webcam for user context switching. |
| `POST` | `/api/knowledge/upload` | `multipart/form-data`. Accepts `.pdf` or `.txt` alongside `user_id` for background Vector DB chunking. |
| `POST` | `/api/knowledge/note` | Accepts simple JSON string notes mapping directly to a user's episodic RAG memory. |

---

## 📌 Troubleshooting

**SpeechBrain Torchaudio Backend Error (Windows)**  
If `libtorchcodec_coreX.dll` fails to load, ensure you run the downgrade specified in our locked environment setup:
```bash
pip uninstall torchcodec
pip install "sentence-transformers<=2.7.0"
```

**Missing GPU Drivers / Slow Processing**  
Ensure your system supports standard CPU AVX offloading if CUDA is not present. The system automatically falls back to CPU parsing natively for MTCNN and SentenceTransformers.

---

## 👨‍💻 Author

**Krupakar Gurije**  
*IT Undergraduate | Aspiring Software Engineer*  

> Built with passion to explore the intersection of explicit human identity, continuous RAG systems, and AI personalization.
