from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)


model = ChatHuggingFace(llm = llm)


chatTemplate = ChatPromptTemplate([
('system', "You are a helpful {domain} expert!!"),
MessagesPlaceholder(variable_name = "history"),
('human', "{input}")
]
)

history = []

chain = chatTemplate | model

while True:
    query = input("You: ")
    if query == "exit":
        break
    result = chain.invoke({
        "domain": "Cricket",
        "history": history,
        "input":query
    })
    history.append(HumanMessage(content = query))
    history.append(AIMessage(content= result.content))
    print(result.content)
print(history)
    
