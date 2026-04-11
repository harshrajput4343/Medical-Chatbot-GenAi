# Medical Chatbot v2 (Production Ready)

Welcome to the upgraded Medical Chatbot. This version features a modular FastAPI backend, a sleek glassmorphic UI, and support for multimodal medical information through Groq and Gemini.

## 🚀 Built With
- **Backend**: FastAPI (Python 3.12+)
- **Storage**: Supabase (Postgres)
- **AI Stack**:
  - Gemini 3 Flash (Primary LLM, configurable to Gemini 2.5 Flash Lite)
  - Pinecone (Vector Search)
  - Groq Llama 3 (Fallback/Multimodal)
  - HuggingFace (Embeddings)
- **Frontend**: TailwindCSS + Alpine.js + Jinja2 (Claude-inspired Premium UI)
- **Features**: Live Technical Pipeline Trace, Source Verification Cards, Instant Light/Dark Mode
- **Deployment**: Docker, Nginx, AWS EC2, GitHub Actions

## 📂 Project Structure
```text
medical-chatbot-v2/
├── app/                  # FastAPI Application
│   ├── main.py           # Core entry point
│   ├── routers/          # Modular API routes
│   └── templates/        # Jinja2 frontend
├── deploy/               # Deployment scripts (EC2, Nginx)
├── static/               # Assets (CSS, JS)
├── src/                  # Core logic helpers (Preserved logic)
└── Dockerfile            # Production-ready image
```

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have Python 3.11+ and Docker installed.

### 2. Configure Environment
Rename `.env.example` to `.env` and fill in your keys:
- `GOOGLE_API_KEY`: For Gemini 1.5 Pro
- `PINECONE_API_KEY`: Vector DB access
- `DATABASE_URL`: Your Supabase Postgres URL
- `GROQ_API_KEY`: For fallback inference

### 3. Initialize Data
Run the following to index your PDFs into Pinecone:
```bash
python store_index.py
```

### 4. Run Locally
```bash
python -m uvicorn app.main:app --reload
```

## 🐳 Docker Deployment
```bash
docker-compose up --build
```

## ☁️ AWS EC2 Deployment
1. SSH into your EC2 instance.
2. Run the setup script:
   ```bash
   bash deploy/setup_ec2.sh
   ```
3. Copy your `.env` to `/opt/medbot/.env`.
4. Enable the service:
   ```bash
   sudo cp deploy/medbot.service /etc/systemd/system/
   sudo systemctl enable --now medbot
   ```

## 🧪 Testing
Run the heartbeat test to verify core functionality:
```bash
pytest tests/test_heartbeat.py
```

---
**Disclaimer**: This chatbot is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
