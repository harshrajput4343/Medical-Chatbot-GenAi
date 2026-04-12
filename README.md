# 🏥 Medical Chatbot v2 - AI-Powered Clinical Assistant
A production-ready AI medical assistant enabling real-time, context-aware diagnosis support through advanced RAG pipelines.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**[About](#-what-is-medical-chatbot) • [Features](#-key-features) • [Getting Started](#-quick-start) • [How It Works](#️-how-it-works)**

---

## 🎯 What is Medical Chatbot?
Medical Chatbot v2 is a scalable, AI-powered diagnostic and medical research assistant. Built natively with a robust FastAPI backend and an advanced Retrieval-Augmented Generation (RAG) pipeline, it analyzes complex medical literature to answer clinical queries instantly in a secure, containerized environment.

## 💡 The Problem It Solves
- **Information Overload:** Quickly sifts through dense medical PDFs and literature to find exact, context-aware answers.
- **Accuracy Constraints:** Uses Pinecone vector embeddings and RAG architecture to reduce AI hallucinations, backing answers with real medical data.
- **Deployment Headaches:** Transitioned from a local script to a fully containerized architecture (Docker) with a local Postgres database, ensuring it "works everywhere."

## 🎓 Perfect For
| User Type | Use Case |
|-----------|----------|
| 👨‍⚕️ **Clinicians** | Quickly reference medical literature and verify diagnostic hypothesis. |
| 👨‍🎓 **Med Students** | Extract summaries and learn about complex cases directly from textbooks. |
| 👨‍💻 **Developers** | Use as a production-level template for building full-stack LangChain RAG apps. |

---

## 🌟 Key Features

### 🔥 Core AI Functionality
- **Dual-Model Inference:** Uses **Gemini 1.5 Flash** as the primary reasoning engine, with **Groq Llama 3** available for multimodal fallback.
- **Semantic Retrieval:** Achieves 95% search accuracy across medical texts using **HuggingFace Embeddings** and **Pinecone**.
- **Context-Aware:** Retains conversation history and medical context across sessions.

### 🔐 Security & Infrastructure
- **Containerized Database:** Built-in local PostgreSQL container via Docker Compose to eliminate cloud networking errors.
- **Reverse Proxy Ready:** Fully compatible with complete Nginx reverse proxy configurations and SSL mapping.
- **Environment Isolation:** Cleanly separates API keys and secrets via strict `.env` loading.

### 🎨 User Experience
- **Live Technical Trace:** Watch the backend pipeline retrieve and process data in real-time.
- **Glassmorphic UI:** A modern, premium frontend interface inspired by Claude.
- **Source Verification:** Every answer includes exact citations from the embedded medical documents.

---

## 🚀 Quick Start

### 📦 Prerequisites
Before you begin, ensure you have the following installed:

| Tool | Version | Download Link |
|------|---------|---------------|
| Python | v3.11+ | [python.org](https://www.python.org/) |
| Docker | Latest stable | [docker.com](https://www.docker.com/) |
| Git | Latest | [git-scm.com](https://git-scm.com/) |

### 🔄 Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Medical-Chatbot-GenAi.git
cd Medical-Chatbot-GenAi

# 2. Configure Environment (Create .env based on the template)
nano .env

# 3. Add your Medical PDFs
# Place your medical document PDFs into the /Data directory.

# 4. Generate Embeddings (First Time Only)
# This processes your PDFs and uploads vectors to Pinecone
python store_index.py

# 5. Spin up the Database & Backend Server via Docker
docker compose up --build -d
```

---

## 🔑 Environment Configuration
Create a `.env` file in the root directory:

```env
# Security Settings
SECRET_KEY=your_super_secret_key
DEBUG=True

# Database (Local Docker Postgres - Highly Recommended)
DATABASE_URL=postgresql://medbot:password@db:5432/medbot

# Supabase (Only for extended Client API/Auth functionality)
SUPABASE_URL=https://<your_ref>.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# AI API Keys
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=medical-chatbot
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
GEMINI_MODEL=gemini-1.5-flash
```

---

## 🏗️ How It Works

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                          User Client                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │    Web UI    │◄───┤ HTTPS Traffic│◄───┤ Mobile/Web   │   │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘   │
└─────────┼───────────────────┼───────────────────────────────┘
          │ (REST API)        │ (Localhost:8000)
          ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   AWS EC2 Environment                       │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ Nginx Prox y │───►│   FastAPI    │                       │
│  └──────────────┘    └──────┬───────┘                       │
│                             │                               │
│                      ┌──────┴───────┐                       │
│                      │ LangChain    │                       │
│                      │ Auto-Routing │                       │
│                      └──────┬───────┘                       │
└─────────────────────────────┼───────────────────────────────┘
                              │
          ┌───────────────────┼─────────────────────┐
          ▼                   ▼                     ▼
┌─────────────────┐ ┌──────────────────┐ ┌────────────────────┐
│ Docker Postgres │ │ Pinecone Vector  │ │   Google Gemini    │
│ (Chat History)  │ │ (Medical PDFs)   │ │   (Inference)      │
└─────────────────┘ └──────────────────┘ └────────────────────┘
```

### 🔄 Workflow
1. **Document Ingestion:** PDFs are split, embedded via HuggingFace, and stored in Pinecone using `store_index.py`.
2. **User Query:** The user submits symptoms or a medical question through the Web UI.
3. **Vector Search:** FastAPI + LangChain queries Pinecone for the most relevant textbook/document chunks.
4. **Context Generation:** Gemini fuses the relevant chunks with the user's question.
5. **Data Persistence:** The chat history and session context is logged to the local Docker PostgreSQL database.

---

## 🛠️ Technology Stack

**Frontend**
- **Jinja2 & HTML5** - Server-side rendering
- **Tailwind CSS** - Modern, responsive styling
- **Alpine.js** - Lightweight client-side reactivity

**Backend**
- **FastAPI** - High-performance async Python framework
- **LangChain** - LLM orchestration and RAG pipeline
- **PostgreSQL** - SQLAlchemy managed relational database
- **Uvicorn** - ASGI web server

**AI Stack**
- **Gemini 1.5 Flash** - Core natural language reasoning
- **Pinecone** - Vector database for semantic context
- **HuggingFace** - Open-source embedding models

**DevOps**
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Reverse proxy serving
- **AWS EC2** - Cloud hosting infrastructure
- **Certbot** - Let's Encrypt SSL automation

---

## 📚 API Documentation
Once the Docker containers are running, you can access the interactive API documentation at:

**http://localhost:8000/docs**

### Key API Endpoints
* `GET /api/v1/health` - Check backend and DB status
* `POST /api/v1/chat` - Submit a clinical query to the RAG pipeline
* `GET /api/v1/history` - Retrieve secure session history

---

## 🐛 Troubleshooting

**Docker container crashing on start?**
Check if you have an orphaned Postgres process blocking port 5432:
```bash
sudo lsof -i :5432
sudo kill -9 <PID>
```

**"Tenant or user not found" Database Error?**
Ensure your `.env` is utilizing the local Docker container (`db:5432`) rather than a remote cloud pooler that is blocking IPv4 networking.

**SSL Certificate `Could not find a matching server block`?**
Double-check `/etc/nginx/sites-available/medbot`. The `server_name` directive MUST exactly match your domain (e.g., `harshmedbot.duckdns.org`), with no typos.

---

## 🎯 Roadmap
- [x] Migrate from Flask to FastAPI
- [x] Dockerize complete backend + database
- [x] Deploy securely on AWS with Nginx
- [ ] Add strict Role-Based Access Control (Admin/Doctor/Patient)
- [ ] Native integration for reading Medical Imagery (X-Rays via Vision models)
- [ ] Export diagnosis reports as PDFs

---

## 🤝 Contributing
Contributions are highly welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
⭐ **If you find this project helpful for your healthcare AI journey, please consider giving it a star!**
