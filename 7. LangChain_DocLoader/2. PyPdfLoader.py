from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("cricket.pdf")

docs = loader.load()

print(docs)

print(docs[0])

print(docs[0].page_content)
print(docs[0].metadata)


