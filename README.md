# RAG with Mistral 7B

This code sets up an interactive chat system:
- It loads documents from a URL, creates a FAISS vector store for them, and initializes a language model.
- RAG for PDF is also available using persist chroma DB
- Users can input messages, and the system responds using the language model.
- The conversation history is stored in a python list, and the loop continues until the user inputs 'exit'.
- Uses mistral-7b-instruct-v0.1.Q5_K_M.gguf for LLM (you need to download it into the repo to use it)
- llama-cpp-python and ctransformers either can be used for LLM inference
- For PDF RAG system, streamlit for UI is also used. For website data RAG, cli is used as the interface.

To simply run the PDFs RAG project:
- Download supported gguf model from HuggingFace. Place the model file in the folder.
- Install packages in your activated environment
```
pip install -r requirement.txt
```
- add your pdfs in data folder in "pdf inference"
- add your huggingface api token in .env file 
- ingest pdfs to chroma db
```
cd "pdf inference"
python ingest.py
```
- after vector db is created, you may run the application:
```
streamlit run app.py
```