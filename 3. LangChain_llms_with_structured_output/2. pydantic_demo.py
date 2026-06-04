from pydantic import BaseModel, Field
from typing import Optional, Literal
#from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI

# hugging face llama does not support function calling using mistral instead

'''
class Student(BaseModel):
    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt =0, lt= 10, default=5, description="A decimal vlaue representing the cgpa of the student")

new_student = {'age':32, "email": 'abc@gmail.com'}
student = Student(**new_student)
'''


class Review(BaseModel):
    key_theme: list[str] = Field(description="Write down all the key themes discussed in the review")
    summary: str = Field(description="A breif summary of the review")
    sentiment: Literal['pos', 'neg'] = Field(description="Return the sentiment of the review neg for negative and pos for positive")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list") 
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list") 
    name: str = Field(description="the name of the product")


model = ChatMistralAI(model= "mistral-small-latest")


structured_model = model.with_structured_output(Review)
result = structured_model.invoke(
"""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
"""
)


print(result)