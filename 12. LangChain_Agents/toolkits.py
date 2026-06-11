from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a+b

@tool
def mult(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

class MathToolkit:
    def get_tools(self):
        return [add,mult]
    

toolkit = MathToolkit()

tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)

