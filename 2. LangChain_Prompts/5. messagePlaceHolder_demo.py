from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are {role}'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
], input_variables = ['role','query'])

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)
model = ChatHuggingFace(llm = llm)

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


role = input("Enter the role of AI: ")
chain = chat_template | model

while True:
    print("You: ")
    query = input()
    if query == "exit":
        break
    result = chain.invoke({
        "role":role,
        "query":query,
        'chat_history':chat_history
    })
    print(result.content)

