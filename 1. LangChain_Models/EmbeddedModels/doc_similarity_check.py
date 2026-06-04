from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()


embedding = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

document = [
    "Subodh wants to have a great body",
    "Namee wants to eat all day",
    "Subodh hate Devanshi!!",
    "Devanshi is a cheater"
]


result = embedding.embed_documents(document)
print(str(result))


query = input("Enter your question:")

question_embed = embedding.embed_query(query)


similarites = cosine_similarity([question_embed],result)[0]

ranked = sorted(zip(similarites, document), reverse = True)


print("Results by similarity:\n")
for score, doc in ranked:
    print(f"Score: {score:.4f} → {doc}")