from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

url = input("Enter the youtube link: ")

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-R1',
    task = 'text-generation'
)

chat_model = ChatHuggingFace(llm = llm)
# deepseek-ai/DeepSeek-R1


loader = YoutubeLoader.from_youtube_url(
    url
)


docs = loader.load()

# print(docs)


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 350,
    chunk_overlap = 20
)

chunks = splitter.split_documents(docs)

vector_store = Chroma(
    embedding_function = embedding_model,
    persist_directory = "chroma_db",
    collection_name = "my_collection_1"
)

# vector_store.add_documents(chunks)


multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(
        search_kwargs= {'k':4}
    ),
    llm = chat_model
)



chat_prompt = ChatPromptTemplate([
    ('system', "you are a chatbot, you will recieve the transcript of a video from youtube and a query from the user on that video's transcript, your task is to response accordingly, if the information is not sufficient to answer that question u may respond i dont know!"),
    MessagesPlaceholder(variable_name="history"),
    ('human', '{context} \n query: {query}')],
    input_variables = ['context', 'query']
)



history = []

parser = StrOutputParser()


chain = chat_prompt | chat_model | parser


while True:
    query = input("You: ")
    if query == "exit":
        break
    history.append(HumanMessage(content= query))
    context = multiquery_retriever.invoke(query)
    context = '\n\n'.join([doc.page_content for doc in context])
    result = chain.invoke({
        "context":context,
        'query':query,
        "history":history
    })
    history.append(AIMessage(content = result))
    print(result)



