# rag_with_mistral_7b

This code sets up an interactive chat system:
- It loads documents from a URL, creates a FAISS vector store for them, and initializes a language model.
- Users can input messages, and the system responds using the language model.
- The conversation history is stored, and the loop continues until the user inputs 'exit'.
- Uses mistral-7b-instruct-v0.1.Q5_K_M.gguf for LLM (you can download it into the repo to use it)