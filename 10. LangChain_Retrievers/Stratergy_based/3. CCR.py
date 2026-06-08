# Contextual Compression Retriever

# it improves the retrieval quality by compressing documents 
# after retrieval keeping only the relavant content based on 
# query
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain_core.documents import Document


# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Sample Documents ───────────────────────────────────────────────────────────
documents = [
    Document(page_content="LangChain is a framework for developing applications powered by large language models. It simplifies chaining multiple LLM calls."),
    Document(page_content="The Eiffel Tower is located in Paris, France. It was constructed in 1889 and is 330 meters tall."),
    Document(page_content="Contextual Compression Retriever compresses retrieved documents using an LLM, keeping only the parts relevant to the query."),
    Document(page_content="FAISS is a library by Facebook AI for efficient similarity search over dense vector embeddings."),
    Document(page_content="Python is a versatile high-level programming language known for readability and broad library support."),
    Document(page_content="RAG (Retrieval-Augmented Generation) grounds LLM responses in retrieved documents to improve factual accuracy."),
]

# ── Step 1: Vector Store → Base Retriever ─────────────────────────────────────
print("Building vector store...")
model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    embedding_function = model,
    persist_directory = "chroma_db",
    collection_name = "my_collection_3"
)

vectorstore.add_documents(documents)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Step 2: LLM via HuggingFaceEndpoint + ChatHuggingFace ─────────────────────
print("Loading LLM...")
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

chat_llm = ChatHuggingFace(llm=llm)

# ── Step 3: Compressor ─────────────────────────────────────────────────────────
compressor = LLMChainExtractor.from_llm(chat_llm)

# ── Step 4: Optional — DocumentCompressorPipeline (chain multiple compressors) ─
# Useful when you want to combine filters + extractors in sequence.
# Here we use just the LLMChainExtractor inside the pipeline.
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[compressor]   # add more transformers here if needed
)

# ── Step 5: ContextualCompressionRetriever ─────────────────────────────────────
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

# ── Step 6: Invoke ─────────────────────────────────────────────────────────────
query = "What is Contextual Compression Retriever in LangChain?"

print(f"\nQuery: {query}")
print("=" * 60)

# Base retriever — returns full document chunks
print("\n[Base Retriever Output — full chunks]")
base_docs = base_retriever.invoke(query)
for i, doc in enumerate(base_docs, 1):
    print(f"\n  Doc {i}: {doc.page_content}")

# Compression retriever — returns only relevant snippets
print("\n[Compression Retriever Output — compressed snippets]")
compressed_docs = compression_retriever.invoke(query)

if compressed_docs:
    for i, doc in enumerate(compressed_docs, 1):
        print(f"\n  Compressed Doc {i}: {doc.page_content}")
else:
    print("  No relevant content extracted after compression.")