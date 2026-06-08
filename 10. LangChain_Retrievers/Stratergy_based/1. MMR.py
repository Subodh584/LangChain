# Maximum marginal relevance

# relavent to query and also different among themselves
# how it works:
# 1. most relavent, 2. most relavent to query but least similar to 1.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()
model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function = model,
    persist_directory = "chroma_db",
    collection_name = "my_collection_1"
)


documents = [
    # Topic 1: Artificial Intelligence
    Document(
        page_content="Artificial intelligence enables machines to perform tasks that normally require human intelligence, such as reasoning, learning, and decision-making.",
        metadata={"topic": "AI"}
    ),
    Document(
        page_content="Machine learning and deep learning are branches of AI that allow computers to learn patterns from data and improve predictions over time.",
        metadata={"topic": "AI"}
    ),

    # Topic 2: Cricket
    Document(
        page_content="Cricket is a popular sport played between two teams. Players score runs by hitting the ball and running between wickets.",
        metadata={"topic": "Cricket"}
    ),
    Document(
        page_content="International cricket tournaments attract millions of viewers, with teams competing in formats such as Test, ODI, and T20 matches.",
        metadata={"topic": "Cricket"}
    ),

    # Topic 3: Space Exploration
    Document(
        page_content="Space agencies launch satellites and spacecraft to study planets, stars, and other celestial objects in the universe.",
        metadata={"topic": "Space"}
    ),
    Document(
        page_content="Modern space missions explore Mars, collect scientific data, and advance our understanding of the solar system.",
        metadata={"topic": "Space"}
    ),

    # Topic 4: Healthy Nutrition
    Document(
        page_content="A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals that support overall health.",
        metadata={"topic": "Nutrition"}
    ),
    Document(
        page_content="Eating nutrient-rich foods such as vegetables, fruits, lean meats, and whole grains helps maintain good health.",
        metadata={"topic": "Nutrition"}
    ),
]


vector_store.add_documents(documents)


retriever = vector_store.as_retriever(
    search_type= 'mmr',
    search_kwargs={
    'k':3,
    'fetch_k':10,
    'lambda_mult':0.5
    }
)


query = "how do machines learn data?"

results = retriever.invoke(query)

print(results)

