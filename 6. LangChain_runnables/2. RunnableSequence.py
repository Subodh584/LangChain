from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    template = "Write a joke on {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "explain this joke {joke}",
    input_variables = ['joke']
)


parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'aalu'})
print(result)
