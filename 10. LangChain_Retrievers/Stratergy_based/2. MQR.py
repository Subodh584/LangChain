# Multy Query Retriever
# A MultiQueryRetriever improves retrieval by generating multiple variations of
# the user's query, searching with each variation, and then combining the results.


from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)

model_1 = ChatHuggingFace(llm = llm)


model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)



from langchain_core.documents import Document

documents = [
    Document(
        page_content="""
        Regular exercise improves cardiovascular health, strengthens muscles,
        increases endurance, and helps maintain a healthy body weight.
        Activities such as walking, running, swimming, and cycling can reduce
        the risk of chronic diseases and improve overall well-being.
        """,
        metadata={"topic": "Exercise"}
    ),

    Document(
        page_content="""
        A balanced diet provides essential nutrients including proteins,
        carbohydrates, fats, vitamins, and minerals. Consuming fruits,
        vegetables, whole grains, and lean protein sources supports
        healthy growth, energy production, and immune function.
        """,
        metadata={"topic": "Nutrition"}
    ),

    Document(
        page_content="""
        Adequate sleep is essential for physical recovery, cognitive
        performance, and emotional well-being. Most adults require
        seven to nine hours of quality sleep each night to maintain
        optimal health and reduce the risk of illness.
        """,
        metadata={"topic": "Sleep"}
    ),

    Document(
        page_content="""
        Mental health is an important component of overall wellness.
        Managing stress through meditation, mindfulness, exercise,
        and social support can improve mood, reduce anxiety, and
        enhance quality of life.
        """,
        metadata={"topic": "Mental Health"}
    ),

    Document(
        page_content="""
        Protein is essential for muscle growth and repair. Common
        protein-rich foods include eggs, chicken breast, fish, dairy
        products, soy chunks, lentils, and whey protein. Adequate
        protein intake supports recovery after exercise.
        """,
        metadata={"topic": "Protein"}
    ),

    Document(
        page_content="""
        Hydration is necessary for maintaining body temperature,
        transporting nutrients, and supporting organ function.
        Drinking sufficient water throughout the day can improve
        physical performance and prevent dehydration.
        """,
        metadata={"topic": "Hydration"}
    ),

    Document(
        page_content="""
        Strength training involves resistance exercises that help
        increase muscle mass, bone density, and metabolic rate.
        Consistent strength training can improve physical function
        and reduce the risk of age-related muscle loss.
        """,
        metadata={"topic": "Strength Training"}
    ),

    Document(
        page_content="""
        The immune system protects the body from infections and diseases.
        Proper nutrition, regular exercise, sufficient sleep, and stress
        management contribute to a stronger immune response and better
        overall health.
        """,
        metadata={"topic": "Immunity"}
    ),
]

vector_store = Chroma(
    embedding_function = model,
    persist_directory = "chroma_db",
    collection_name = "my_collection_2"
)

vector_store.add_documents(documents)

similarity_retriever = vector_store.as_retriever(
    search_type = 'similarity', search_kwargs = {'k':5}
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {'k':5}),
    llm = model_1
)

query = "what should i do to improve my health?"


result1 = similarity_retriever.invoke(query)

result2 = multiquery_retriever.invoke(query)

print(result1)

print(result2)