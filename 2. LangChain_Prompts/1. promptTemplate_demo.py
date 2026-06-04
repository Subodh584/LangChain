from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task= "text-generation"
)
model = ChatHuggingFace(llm = llm)

paper_input = input("Enter the paper Name: ")
style_input = input("Enter the style: ")
length_input = input("Enter the length in words: ")

template = PromptTemplate(
    template = """
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input} words
Mathematical Details:
Include relevant mathematical equations if present in the paper.
Explain the mathematical concepts using simple, intuitive code snippets where applicable.
Analogies:
Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""", 
input_variables= ['paper_input', 'style_input', 'length_input'],
validate_template=True
)

'''
prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

result = model.invoke(prompt)
'''

#OR

chain = template | model
result = chain.invoke(
    {
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
}
)






print(result.content)
