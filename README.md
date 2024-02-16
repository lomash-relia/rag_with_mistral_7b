# rag_with_mistral_7b

This code sets up an interactive chat system:
- It loads documents from a URL, creates a FAISS vector store for them, and initializes a language model.
- RAG for PDF is also available using persist chroma DB
- Users can input messages, and the system responds using the language model.
- The conversation history is stored in a python list, and the loop continues until the user inputs 'exit'.
- Uses mistral-7b-instruct-v0.1.Q5_K_M.gguf for LLM (you can download it into the repo to use it)
- llama-cpp-python and ctransformers both can be used for LLM inference
- For PDF RAF system, streamlit for UI is also used. For website data RAG, cli is used as the interface.