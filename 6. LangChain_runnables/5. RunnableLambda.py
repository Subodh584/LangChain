# RunnableeLambda is used to convert
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

print(runnable_word_counter.invoke("Hi there how are you"))





load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'write a joke on {topic} \n only the joke',
    input_variables = ['topic']
)

parser = StrOutputParser()


sequence_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLambda(word_counter)
})

final_chain = sequence_chain | parallel_chain

result = final_chain.invoke({"topic": "Elephant and ant"})

print(result)