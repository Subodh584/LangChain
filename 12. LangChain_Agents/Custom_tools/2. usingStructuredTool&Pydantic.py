from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required = True, description= 'the first number to add')
    b: int = Field(required = True, description= "the second number to add")

def multiply_func(a:int, b: int) -> int:
    return a * b


multiply_tool = StructuredTool.from_function(
    func = multiply_func,
    name = "multiply",
    description = "Multiply two numbers",
    args_schema = MultiplyInput
)


result = multiply_tool.invoke({
    "a":3,
    "b":15
})


print(result)

