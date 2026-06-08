# if else of langchain
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

class Review(BaseModel):
    sentiment: Literal["pos", "neg"] = Field(description="resturn 'pos' is sentiment is positive else return 'neg'")

parser_pydantic = PydanticOutputParser(pydantic_object = Review)

prompt = PromptTemplate(
    template = "Analyse this review and find out the sentiment of it \n {review}, {format_instructions}",
    input_variables = ['review'],
    partial_variables = {'format_instructions': parser_pydantic.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template = "Write an appropreate response to this positive feedback \n {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropreate response to this negative feedback \n {feedback}",
    input_variables = ['feedback']
)


sentiment_chain = prompt | model | parser_pydantic


review = """I purchased the SmartX Pro Wireless Earbuds hoping for a premium audio experience, but unfortunately, they failed to meet even basic expectations.

The sound quality was disappointing, with weak bass, muddy vocals, and noticeable distortion at higher volumes. The battery life was significantly lower than advertised, requiring frequent charging throughout the day. Connectivity was another major issue; the earbuds frequently disconnected from my phone and often struggled to reconnect automatically.

The build quality also felt cheap despite the product being marketed as a premium device. After only a few weeks of normal use, the charging case developed a loose hinge and the earbuds showed visible wear. The touch controls were inconsistent and often registered accidental inputs.

Customer support was equally frustrating. Responses were slow, and the solutions provided did not resolve the issues I experienced. For the price, I expected a much more reliable and polished product.

Overall, the SmartX Pro Wireless Earbuds offered poor performance, questionable durability, and an unsatisfactory user experience. I would not recommend this product and would suggest looking at other alternatives in the same price range."""




parser = StrOutputParser()

conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == 'pos', prompt2 | model | parser),
    (lambda x: x.sentiment == 'neg', prompt3 | model | parser),
    RunnableLambda(lambda x: "Sentiment not detected")
)



final_chain = sentiment_chain | conditional_chain 
result = final_chain.invoke({
    'review': review
})


print(result)