from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = "Generate a {length} report on {topic}",
    input_variables = ['length', 'topic'] 
)
parser = StrOutputParser()

report_chain = prompt1 | model | parser


def length_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    "report": RunnablePassthrough(),
    "length": RunnableLambda(length_counter)
})

len_and_report_chain = report_chain | parallel_chain




prompt2 = PromptTemplate(
    template = "Summarize this report: {report}",
    input_variables = ["report"]
)


def extract_report(x):
    return {'report': x["report"]}

conditional_chain = RunnableBranch(
    (lambda x: x['length'] > 500, extract_report | prompt2 | model | parser),
    (lambda x: {"report": x["report"]})
)


final_chain = len_and_report_chain | conditional_chain

result = final_chain.invoke({
    "length": "super long",
    "topic": "Transformers in deep learning"                          
})

print(result)