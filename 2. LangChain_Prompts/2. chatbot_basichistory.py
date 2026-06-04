from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm= llm)

chat_history = []


x = True
while x != False:
    query = input("You: ")
    chat_history.append("user: " + query)
    if query == "exit":
        x = False
        break
    result = model.invoke(chat_history)
    chat_history.append("AI: " + result.content)
    print(result.content)

print(chat_history)