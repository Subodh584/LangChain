from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import re
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

chatPrompt = ChatPromptTemplate([
    ('system', "You are a chatbot, you will be given a scraped data, and then some questions will be asked and your job is to answer every question in a professional manner"),
    ('system', "here is the data: \n {scraped_data}"),
    MessagesPlaceholder(variable_name = "chat_history")
    ],
    input_variables = ['scraped_data']
)
loader = WebBaseLoader(web_paths=["https://towardsdatascience.com/stop-using-llms-like-giant-problem-solvers/"])
docs = loader.load()
def clean_text(text):
    # Replace multiple whitespace/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

for doc in docs:
    doc.page_content = clean_text(doc.page_content)

print(len(docs))
print(docs[0].page_content)




chat_history = []
parser = StrOutputParser()

chain = chatPrompt | model | parser

while True:
    print("You: ")
    query = input()
    if query == 'exit':
        break
    chat_history.append(HumanMessage(content = query))
    result = chain.invoke({
        'chat_history': chat_history,
        'scraped_data': docs[0].page_content
    })
    chat_history.append(AIMessage(content = result))
    print(result)

