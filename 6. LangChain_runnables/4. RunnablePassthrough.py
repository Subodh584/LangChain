from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'write a joke on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'explain this joke {joke}',
    input_variables = ['joke']
)

parser = StrOutputParser()
passthrough = RunnablePassthrough()

joke_chain = prompt1 | model | parser 

parallel_chain = RunnableParallel({
    'joke': passthrough,
    'explaination': prompt2 | model | parser
})

final_chain = joke_chain | parallel_chain

result = final_chain.invoke({"topic": "Elephant and ant"})

print(result)




# "input" --> RunnablePassthrough() --> "input"




# usecase - Eg. lets say we have a chatModel in which we are giving a topic and asking a it make a joke on this topic, and then sending that joke to the same or diff chatmodel and getting the explaination,then in thiss case we only get one output and that is the joke explaination alone so i order to get the joke and explaination both we need this runnable, it can be used in a parallel chain