from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever(top_k_results = 2, lang="en")


query = "SRM institute of Science and technology"


docs = retriever.invoke(query)

print(docs)