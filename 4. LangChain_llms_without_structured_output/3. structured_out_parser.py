from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser        # ✅ from langchain_core
from langchain.output_parsers import StructuredOutputParser, ResponseSchema  # ✅ from langchain
load_dotenv()

model = ChatMistralAI(model="mistral-small-latest")

schema = [
    ResponseSchema(name='fact_1', description="Fact1 about the topic"),
    ResponseSchema(name='fact_2', description="Fact2 about the topic"),
    ResponseSchema(name='fact_3', description="Fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "black holes"})
print(result)