from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = """
The human heart is a muscular organ that pumps blood throughout the body. 
It beats approximately 100,000 times per day, circulating oxygen and nutrients to tissues and organs. 
The heart has four chambers: the left and right atria, and the left and right ventricles. 
Cardiovascular diseases remain the leading cause of death worldwide, making heart health a critical medical priority.
Regular exercise, a balanced diet, and avoiding smoking significantly reduce the risk of heart disease.
Doctors often recommend monitoring blood pressure and cholesterol levels to maintain a healthy heart.

The Amazon rainforest spans over 5.5 million square kilometers across South America. 
It is home to an estimated 10% of all species on Earth, making it the most biodiverse region on the planet. 
The forest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide. 
Deforestation driven by agriculture, logging, and urban expansion threatens this vital ecosystem every year. 
Indigenous communities who have lived in the Amazon for thousands of years are also displaced by these activities. 
Conservation efforts led by governments and NGOs aim to protect remaining forest areas through legal protections and sustainable land-use practices.

Quantum computing is a rapidly evolving field that leverages the principles of quantum mechanics to process information. 
Unlike classical computers that use bits represented as 0s and 1s, quantum computers use qubits that can exist in multiple states simultaneously. 
This property, known as superposition, allows quantum computers to solve certain problems exponentially faster than traditional machines. 
Companies like IBM, Google, and startups worldwide are racing to build reliable quantum hardware. 
Quantum computing holds promise for fields such as cryptography, drug discovery, financial modeling, and artificial intelligence.

The Renaissance was a cultural and intellectual movement that began in Italy during the 14th century. 
It marked a transition from the medieval period to modernity, emphasizing humanism, art, and scientific inquiry. 
Artists like Leonardo da Vinci and Michelangelo produced iconic works that defined the era. 
The invention of the printing press by Johannes Gutenberg accelerated the spread of Renaissance ideas across Europe. 
This period laid the foundation for the Scientific Revolution and the Enlightenment that followed in later centuries.

Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. 
Algorithms identify patterns in large datasets and improve their performance over time through experience. 
Supervised learning, unsupervised learning, and reinforcement learning are the three primary types of machine learning approaches. 
Applications range from image recognition and natural language processing to fraud detection and recommendation systems. 
The rapid growth of data and computational power has made machine learning one of the most transformative technologies of the modern era.
"""



text_splitter = SemanticChunker(
    embedding, breakpoint_threshold_type = "standard_deviation",
    breakpoint_threshold_amount = 1
)


docs = text_splitter.create_documents([text])

print(len(docs))
print(docs) 