from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

chat_template = ChatPromptTemplate([
    ('system','You are a report generator ai and u will be given a topic and your task is to make a detailed report for that topic'),
    ('human', 'your topic is {topic}'),
], input_variables= ['topic'])

chat_template2 = ChatPromptTemplate([
    ('system','You are a report summarizer ai and u will be given a detailed report and u have to summarize it in 50 to 100 words'),
    ('human', '{report}'),
], input_variables= ['report'])

model = ChatMistralAI(model="mistral-small-latest")


#without string output parser
'''
prompt1 = chat_template.invoke({"topic": "transformers in deeplearning"})
result = model.invoke(prompt1)
prompt2 = chat_template2.invoke({"report":result.content})
print(result.content)
result2 = model.invoke(prompt2)
print(result2.content)
'''

#with string output parser
parser = StrOutputParser()

chain = chat_template | model | parser | chat_template2 | model | parser

result = chain.invoke({'topic': "Transformeers in deep learning"})
print(result)

