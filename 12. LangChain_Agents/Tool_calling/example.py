from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()


@tool
def mult(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b


llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)


model_with_tool = model.bind_tools([mult])

query = HumanMessage("can u multiply 231546982 and 1046549980")
messages = [query]

result = model_with_tool.invoke(messages)

messages.append(result)



# res1 = result.tool_calls[0]['args']
# print(mult.invoke(res1)) --> returns 

res2 = result.tool_calls[0]
tool_result = mult.invoke(res2)



messages.append(tool_result)

result_final = model_with_tool.invoke(messages)


print(result_final.content)

# tool calling doesnt mean that our llm is calling the 
# function, it simply suggests the input arguments




