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


prompt1 = PromptTemplate(
    template = """
Create a detailed report on the topic {topic}
""", input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = """
here is a detailed report \n
{report}
make a short summary for this report.
"""
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({
    "topic": "SRM Institute Of Science and technology"
})

print(result)

chain.get_graph().print_ascii()