from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template= """
you are an expert in {domain} tell me about {topic}
""", input_variables = ['domain','topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({
    "domain": "Badminton",
    "topic":"Most matches won by which player?"
})

print(result)

chain.get_graph().print_ascii()