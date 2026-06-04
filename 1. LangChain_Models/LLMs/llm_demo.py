#from langchain_openai import OpenAI
#from langchain_mistralai import MistralAI

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()



#llm = MistralAI(model = 'gpt-3.5-turbo-instruct')
#llm = OpenAI(model = 'open-mistral-7b')

llm = ChatMistralAI(model = 'mistral-small-latest')



result = llm.invoke("What is the capital of India in short")

print(result.content)





# i dont have api key so i cannot use chatGPT, and also mistral
# does not support llms anymore...