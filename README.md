# 8-Day Applied Generative AI & Agentic Systems Bootcamp

A production-oriented GenAI engineering portfolio built with **LangChain**, **Google Gemini**, **Groq**, **FAISS**, **Pydantic**, and **Tavily**. This repository documents an 8-day progressive curriculum covering LLM foundations, conversational memory optimization, document ingestion, vector retrieval (RAG with MMR), LCEL pipelines, and autonomous multi-tool AI agents.

---

## System Architecture

```mermaid
flowchart TD
    subgraph D1["Day 1 & 2: Foundations & Memory"]
        A[User Query] --> B[LLM API / Cloud Workspace]
        B --> C[Sliding-Window Memory<br/>System + Last 6 Messages]
    end

    subgraph D2["Day 3 & 4: Ingestion & Vector Storage"]
        D[PDF / Raw Docs] --> E[PyPDFLoader]
        E --> F[RecursiveCharacterTextSplitter<br/>chunk_size=800, overlap=150]
        F --> G[Vector Store<br/>FAISS / In-Memory Index]
    end

    subgraph D3["Day 5 & 6: RAG & LCEL Pipelines"]
        G --> H[MMR Retriever<br/>λ=0.5 Relevance vs Diversity]
        H --> I[LCEL RunnableSequence<br/>Parallel Evaluation & Citations]
    end

    subgraph D4["Day 7 & 8: Agentic AI & Tool Routing"]
        I --> J[Autonomous Agent<br/>create_agent Reasoning Loop]
        J --> K[PDF Knowledge Base Tool]
        J --> L[OpenWeather Live API Tool]
        J --> M[Tavily Search Engine Tool]
    end

    J --> N[Structured / Guardrailed Final Response]
```
