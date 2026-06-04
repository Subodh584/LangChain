from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
model = ChatMistralAI(model = 'mistral-small-latest', 
temperature = 0.9, max_tokens = 10)
result = model.invoke("who are u?")
print(result.content)


'''
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()
model = CharAnthropic(model = "claude-3-5-sonnet-20241022")
model.invoke("who are u?")
print(result.contnet)
'''



'''
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model = "")
result = model.invoke("Who are u?")
print(result.content)
'''


