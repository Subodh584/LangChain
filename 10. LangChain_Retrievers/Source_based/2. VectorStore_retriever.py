from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document



from langchain_core.documents import Document

documents = [
    Document(
        page_content="""
        Artificial Intelligence (AI) is a branch of computer science focused on
        creating systems that can perform tasks typically requiring human intelligence.
        These tasks include learning, reasoning, problem-solving, language understanding,
        and decision-making. Modern AI systems often rely on machine learning and deep
        learning techniques, where models are trained on large datasets to recognize
        patterns and make predictions. AI is widely used in recommendation systems,
        virtual assistants, autonomous vehicles, healthcare diagnostics, and financial
        analysis. While AI offers significant benefits in automation and efficiency,
        it also raises concerns regarding ethics, privacy, bias, and job displacement.
        """
    ),
    
    Document(
        page_content="""
        The Amazon Rainforest is the world's largest tropical rainforest, covering
        millions of square kilometers across South America. It is home to an incredible
        diversity of plant and animal species, many of which are found nowhere else on
        Earth. The rainforest plays a crucial role in regulating the global climate by
        absorbing carbon dioxide and producing oxygen. Indigenous communities have lived
        in the Amazon for thousands of years, developing unique cultures and knowledge
        of the ecosystem. However, deforestation, mining, and agricultural expansion
        continue to threaten the region's biodiversity and environmental importance.
        """
    ),
    
    Document(
        page_content="""
        Cricket is one of the most popular sports in the world, particularly in countries
        such as India, Australia, England, Pakistan, and South Africa. The game is played
        between two teams of eleven players and involves batting, bowling, and fielding.
        Various formats exist, including Test cricket, One Day Internationals (ODIs),
        and Twenty20 (T20) matches. The sport requires a combination of technical skill,
        strategy, teamwork, and mental resilience. Major tournaments such as the Cricket
        World Cup and popular leagues like the IPL attract millions of viewers globally.
        """
    ),
    
    Document(
        page_content="""
        Space exploration refers to the investigation of outer space using telescopes,
        satellites, robotic probes, and crewed spacecraft. Since the launch of the first
        artificial satellite, Sputnik 1, humanity has achieved numerous milestones,
        including landing astronauts on the Moon and sending probes to distant planets.
        Space missions contribute to scientific understanding of the universe, planetary
        systems, and the origins of life. Technological innovations developed for space
        programs often find applications on Earth in communication, navigation, medicine,
        and engineering. Future goals include establishing lunar bases and sending humans
        to Mars.
        """
    ),
    
    Document(
        page_content="""
        Healthy nutrition is essential for maintaining physical and mental well-being.
        A balanced diet typically includes adequate amounts of protein, carbohydrates,
        healthy fats, vitamins, minerals, and water. Proper nutrition supports muscle
        growth, immune function, cognitive performance, and overall energy levels.
        Consuming a variety of fruits, vegetables, whole grains, lean proteins, and
        healthy fats can help reduce the risk of chronic diseases such as diabetes,
        heart disease, and obesity. Regular physical activity combined with healthy
        eating habits forms the foundation of long-term health and fitness.
        """
    )
]


model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# creating vectorstore and storing from the documents
# vector_store = Chroma(
#     documents = documents,
#     embedding = model,
#     collection_name = "my_collection"
# )


vector_store = Chroma(
    embedding_function = model,
    persist_directory = "chroma_db",
    collection_name = "my_collection"
)


vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_kwargs={'k':2})
query = "cricket"
results = retriever.invoke(query)
print(results)