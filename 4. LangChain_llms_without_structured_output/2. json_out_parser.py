from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
load_dotenv()
# Define your schema
class Review(BaseModel):
    name: str = Field(description="Product name")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment")
    pros: Optional[list[str]] = Field(description="List of pros")

# JsonOutputParser reads your Pydantic model and generates format instructions
parser = JsonOutputParser(pydantic_object=Review)

# Inject format instructions into the prompt
prompt = ChatPromptTemplate([
    ('system', 'Extract review information. {format_instructions}'),
    ('human', '{review}')
],partial_variables={"format_instructions":parser.get_format_instructions()})
# partial() pre-fills a variable so you dont have to pass it every invoke

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)

chain = prompt | model | parser

result = chain.invoke({"review": "Samsung S24 is great but expensive..."})
print(result)        # dict
print(result["name"])  # "Samsung Galaxy S24"