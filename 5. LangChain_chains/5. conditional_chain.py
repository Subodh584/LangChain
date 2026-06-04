from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

parser = StrOutputParser()
model = ChatMistralAI(model = "mistral-small-latest")

class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description= "if the sentiment is positive reply 'positive' else 'negative'")


structured_model = model.with_structured_output(FeedBack)


prompt1 = PromptTemplate(
    template = """
I recently had the opportunity to use an Apple device, and the experience has been excellent. The device stands out for its premium design, smooth performance, and user-friendly interface. Everything feels well-optimized, from opening applications to multitasking between different tasks.

One of the most impressive aspects is the reliability and stability of the operating system. Applications run smoothly with minimal lag, and the overall user experience feels polished and intuitive. The display quality is outstanding, offering vibrant colors, sharp details, and an immersive viewing experience for both work and entertainment.

The battery life is another strong feature, allowing extended usage throughout the day without frequent charging. The device also integrates seamlessly with other Apple products, creating a convenient and productive ecosystem that enhances everyday tasks.

Security and privacy features provide additional confidence, ensuring that personal information remains protected. Regular software updates further improve performance and introduce useful features, demonstrating Apple's commitment to long-term support.

Overall, the Apple device delivers a premium experience through its combination of elegant design, powerful performance, excellent battery life, high-quality display, and seamless ecosystem integration. It is a reliable and enjoyable device that meets both personal and professional needs effectively.
"""
)


classifier_chain = prompt1 | structured_model
result = classifier_chain.invoke({})
print(result)

prompt2 = PromptTemplate(
    template = "Write an appropreate response to this positive feedback \n {feedback}",
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropreate response to this negative feedback \n {feedback}",
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x['feedback'] == 'positive', prompt2 | model | parser),
    (lambda x: x['feedback'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

def format_for_branch(input):
    result = classifier_chain.invoke({})
    return {
        "feedback": result.sentiment,
    }




merge_chain = RunnableLambda(format_for_branch) | branch_chain


result = merge_chain.invoke({})

print(result)




