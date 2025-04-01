from mcp.server.fastmcp import FastMCP 
import types


mcp =FastMCP("Math") 

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


@mcp.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="https://zvahitrpogaxxlgijefl.supabase.co",
            name="data",
            mimetype="application/json"
        )
    ]
    
@mcp.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    if str(uri) == "":
        log_

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

if __name__ =="__main__":
    mcp.run(transport="stdio")