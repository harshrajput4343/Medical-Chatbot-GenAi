from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from .rag_init import rag_manager
from ..config import settings
import httpx
import time

class ChatService:
    def __init__(self):
        self.rag = rag_manager
        self.system_prompt = (
            "You are a Senior Medical Information Specialist.\n"
            "Instruction: Provide a structured response with the following sections:\n"
            "(Summary | Key Points | When to See a Doctor | Disclaimer)\n\n"
            "Always cite which document/source the info came from.\n"
            "If no context found, say so clearly and suggest seeing a doctor.\n"
            "Max 5 sentences per section.\n\n"
            "Context: {context}"
        )

    async def get_response(self, question: str, is_multimodal: bool = False):
        if is_multimodal and settings.GROQ_API_KEY:
            return await self._get_groq_fallback(question)
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
            ])

            combine_docs_chain = create_stuff_documents_chain(self.rag.get_llm(), prompt)
            retrieval_chain = create_retrieval_chain(self.rag.get_retriever(), combine_docs_chain)
            
            start_time = time.time()
            response = retrieval_chain.invoke({"input": question})
            end_time = time.time()

            sources = []
            for doc in response.get("context", []):
                sources.append(doc.metadata.get("source", "Unknown Source"))

            return {
                "answer": response["answer"],
                "sources": list(set(sources)),
                "response_time_ms": int((end_time - start_time) * 1000)
            }
        except Exception as e:
            if settings.GROQ_API_KEY:
                return await self._get_groq_fallback(question)
            raise e

    async def _get_groq_fallback(self, question: str):
        # Fallback to Groq if Gemini fails or multimodal is requested
        # For simplicity, we use Llama 3 70B on Groq for fast inference
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a Medical Assistant. Provide clear, safe medical information."},
                    {"role": "user", "content": question}
                ]
            }
            response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            res_json = response.json()
            return {
                "answer": res_json["choices"][0]["message"]["content"],
                "sources": ["Groq Fallback (No local context)"],
                "response_time_ms": 0
            }

chat_service = ChatService()
