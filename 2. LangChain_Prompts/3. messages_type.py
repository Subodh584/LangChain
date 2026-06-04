from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpfull assistant")
]

while True:
    print("You: ")
    query = input()
    messages.append(HumanMessage(content = query))
    if query == "exit":
        break
    result = model.invoke(messages)
    messages.append(AIMessage(content = result.content))
    print("AI: \n" + result.content)

print(messages)
