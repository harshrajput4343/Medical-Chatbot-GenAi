<div align="center">

# 🏥 Medical Chatbot with GenAI

### An intelligent medical assistant powered by LLMs, LangChain, Pinecone & Flask

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.26-green.svg)](https://langchain.com/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-red.svg)](https://flask.palletsprojects.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple.svg)](https://www.pinecone.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#%EF%B8%8F-aws-cicd-deployment)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Architecture & Workflow](#-architecture--workflow)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#%EF%B8%8F-installation)
- [Run Locally](#-run-locally)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [AWS CI/CD Deployment](#%EF%B8%8F-aws-cicd-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About the Project

This Medical Chatbot is an AI-powered conversational assistant designed to answer medical-related questions. It leverages **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, context-aware responses by combining:

- **Vector Search**: Using Pinecone for semantic similarity search
- **LLM Integration**: Google Gemini for natural language understanding and generation
- **Document Processing**: PDF medical documents as the knowledge base

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **RAG Architecture** | Combines retrieval and generation for accurate responses |
| 📚 **PDF Knowledge Base** | Extracts and indexes medical documents |
| 🔍 **Semantic Search** | Uses vector embeddings for relevant context retrieval |
| 💬 **Interactive Chat UI** | Clean Flask-based web interface |
| ☁️ **Cloud Ready** | Docker support with AWS deployment pipeline |
| 🔄 **CI/CD Pipeline** | Automated deployment with GitHub Actions |

---

## 🏗 Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEDICAL CHATBOT WORKFLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Medical PDFs   │
                              │   (Data folder)  │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION PIPELINE                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │  PDF Loader │ -> │ Text Split  │ -> │  Embedding  │ -> │   Pinecone   │  │
│  │ (PyPDFLoader)   │ (500 chunks) │    │ (MiniLM-L6) │    │ Vector Store │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │    User Query    │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RAG PIPELINE                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │    Query    │ -> │  Retriever  │ -> │   Context   │ -> │    Gemini    │  │
│  │  Embedding  │    │  (Top 3)    │    │  + Prompt   │    │   Response   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Chat Response  │
                              └──────────────────┘
```

### How It Works

1. **Document Processing** (`store_index.py`)
   - Load PDF documents from the `Data/` folder
   - Split documents into 500-character chunks with 20-character overlap
   - Generate embeddings using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`
   - Store vectors in Pinecone index

2. **Query Processing** (`app.py`)
   - User submits a question via the web interface
   - Query is converted to embedding vector
   - Pinecone retrieves top 3 most similar chunks
   - Context is combined with the system prompt
   - Google Gemini generates a concise, contextual response

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10 |
| **Framework** | Flask 3.1.1 |
| **LLM** | Google Gemini 2.5 Pro |
| **Orchestration** | LangChain 0.3.26 |
| **Vector Database** | Pinecone |
| **Embeddings** | HuggingFace Sentence Transformers |
| **Containerization** | Docker |
| **Cloud** | AWS (EC2, ECR) |
| **CI/CD** | GitHub Actions |

---

## 📁 Folder Structure

```
Medical-Chatbot-GenAi/
│
├── 📄 app.py                    # Main Flask application
├── 📄 store_index.py            # Script to index documents to Pinecone
├── 📄 setup.py                  # Package setup configuration
├── 📄 requirements.txt          # Python dependencies
├── 📄 Dockerfile                # Docker configuration
├── 📄 .env                      # Environment variables (create this)
│
├── 📁 src/                      # Source code package
│   ├── 📄 __init__.py           # Package initializer
│   ├── 📄 helper.py             # Utility functions (PDF loading, embeddings)
│   └── 📄 prompt.py             # System prompt template
│
├── 📁 Data/                     # Medical documents (PDFs)
│   └── 📄 Medical_book.pdf      # Knowledge base document
│
├── 📁 templates/                # HTML templates
│   └── 📄 chat.html             # Chat interface
│
├── 📁 static/                   # Static assets
│   └── 📄 style.css             # Stylesheet
│
├── 📁 research/                 # Jupyter notebooks for experimentation
│   └── 📄 trials.ipynb          # Development trials
│
└── 📁 .github/workflows/        # CI/CD pipeline configuration
    └── 📄 main.yml              # GitHub Actions workflow
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- Git
- Pinecone account
- Google AI API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/harshrajput4343/Medical-Chatbot-GenAi.git
cd Medical-Chatbot-GenAi
```

### Step 2: Create Conda Environment

```bash
conda create -n medchatbot python=3.10 -y
conda activate medchatbot
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

| Variable | Description | Required |
|----------|-------------|----------|
| `PINECONE_API_KEY` | Pinecone vector database API key | ✅ |
| `OPENAI_API_KEY` | OpenAI API key (optional fallback) | ❌ |
| `GOOGLE_API_KEY` | Google Gemini API key | ✅ |

---

## 🚀 Run Locally

### Step 1: Index Your Documents

First, ensure your medical PDFs are in the `Data/` folder, then run:

```bash
python store_index.py
```

This will:
- Load all PDFs from `Data/` folder
- Split them into chunks
- Generate embeddings
- Store vectors in Pinecone

### Step 2: Start the Application

```bash
python app.py
```

### Step 3: Access the Chatbot

Open your browser and navigate to:

```
http://localhost:8080
```

Or access via your local IP:

```
http://127.0.0.1:8080
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Renders the chat interface |
| `POST` | `/get` | Processes user message and returns AI response |

### Example Request

```bash
curl -X POST http://localhost:8080/get \
  -d "msg=What are the symptoms of diabetes?"
```

---

## ☁️ AWS CI/CD Deployment

### Overview

Deploy the application to AWS using Docker, ECR, and EC2 with GitHub Actions for continuous deployment.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │ --> │   Docker    │ --> │     ECR     │ --> │     EC2     │
│    Push     │     │    Build    │     │    Push     │     │   Deploy    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Step 1: AWS Setup

1. **Login to AWS Console**

2. **Create IAM User** with the following policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`

3. **Create ECR Repository**
   ```
   Save the URI: <account-id>.dkr.ecr.<region>.amazonaws.com/medicalbot
   ```

4. **Create EC2 Instance** (Ubuntu)

### Step 2: Configure EC2

SSH into your EC2 instance and run:

```bash
# Update packages
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 3: Configure Self-Hosted Runner

1. Go to Repository **Settings** → **Actions** → **Runners**
2. Click **New self-hosted runner**
3. Choose **Linux** and follow the commands

### Step 4: Setup GitHub Secrets

Add these secrets in your repository settings:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g., `us-east-1`) |
| `ECR_REPO` | ECR repository name |
| `PINECONE_API_KEY` | Pinecone API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google API key |

### Step 5: Deploy

Push to the main branch to trigger automatic deployment:

```bash
git push origin main
```

---

## 🐳 Docker

### Build the Image

```bash
docker build -t medical-chatbot .
```

### Run the Container

```bash
docker run -p 8080:8080 --env-file .env medical-chatbot
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com/) - LLM orchestration framework
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Google Gemini](https://ai.google.dev/) - Large Language Model
- [HuggingFace](https://huggingface.co/) - Sentence Transformers

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

Made with passion by [Harsh Rajput](https://github.com/harshrajput4343)

</div>
