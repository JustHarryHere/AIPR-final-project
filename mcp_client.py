import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_DIR = Path(__file__).resolve().parent
SERVERFIFE = PROJECT_DIR /"mcp_server" / "server.py"

def display_tools(tools):
    """Display tools discovered dynamically from the MCP Server"""
    if not tools:
        print("Server không cung cáp tool nào ")
        return

    print("\n=== Tools discovered from MCP Server ===")

    for index, tool in enumerate(tools, start=1):
        print(f"\{index}.       Tool name: {tool.name}")
        print(f"    Desciption: {getattr(tool,
                                         'description', None) or '(none)'}")
        print(
            " input Schema: "
            +json.dumps(
                getattr(tool,"inputSchema", {},)
            )
        )