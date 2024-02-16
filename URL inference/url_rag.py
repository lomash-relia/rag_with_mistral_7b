# from ctransformers import AutoModelForCausalLM

from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser,JsonOutputParser,StrOutputParser
# from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# embeddings
def initialize_embeddings():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embedding

# URL docs loader
def load_docs(url):
    loader = WebBaseLoader(url,requests_per_second=1)
    docs = loader.load()
    splitter= RecursiveCharacterTextSplitter(chunk_size = 20,chunk_overlap = 5)
    docs = splitter.split_documents(docs)
    return docs
 
# docA = Document(page_content="My name is Lomash")
def create_db(url):
    embedding = initialize_embeddings()
    docs = load_docs(url)
    print(f"docs length: {len(docs)}")
    vectoreStore = FAISS.from_documents(docs,embedding=embedding)
    print(vectoreStore)
    return vectoreStore

def create_chain(vectoreStore: FAISS):
    config={'context_length':1024,
            'temperature':0.02}
    model=r"LLM model\mistral-7b-instruct-v0.1.Q5_K_M.gguf"
    llm = CTransformers(model=model,
                        model_type="mistral",
                        config=config)
    print(llm)

    # response = llm.batch(["Hello how are you?","Write a poem on Cosmos"])

    # response = llm.stream("Write a poem on cosmos.")
    # for chunk in response:
    #     print(chunk,end="",flush=True)

    # Prompt
    # prompt = ChatPromptTemplate.from_template("""
    #                                         Answer the user's question. Be precise.
    #                                         Context: {context}
    #                                         Question: {input}
    #                                         """)

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are 'caption for social media' generating AI assistant.Generate caption based on following {context}."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ])

    # parser = CommaSeparatedListOutputParser()

    # Chain
    # chain = prompt | llm | parser
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    retriever = vectoreStore.as_retriever(search_kwargs={"k":3})
    retrieval_chain = create_retrieval_chain(retriever,
                                   chain)
    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input":question,
        "chat_history":chat_history
        })
    return response
    
if __name__ == "__main__":
    # url = "https://machinelearningmastery.com/start-here/#code_algorithms"
    url = "https://napoleoncat.com/blog/instagram-captions/"
    vectoreStore = create_db(url)
    chain = create_chain(vectoreStore)
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = process_chat(chain, user_input,chat_history)
        # chat_history.append(HumanMessage(content=user_input))
        # chat_history.append(AIMessage(content=str(response)))
        
        print(response['answer'])