from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task=""
)

model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate(
    [
        ('system', 'You are a helpful {domain} expert'),
        ('human', 'Explain in simple termns, what is {topic}')
    ], input_variables = ['domain', 'topic']
)

'''
prompt = chat_template.invoke({
    'domain':"cricket",
    'topic':"dusra"
})
'''

#OR


chain = chat_template | model

result = chain.invoke({
    'domain':"cricket",
    'topic':"dusra"
})

print(result.content)