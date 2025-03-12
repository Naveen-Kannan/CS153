#!/usr/bin/env python3
"""
MCP GitHub Client Class - Connects and provides an MCP session
"""

import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self, token=None):
        """
        Initialize the MCP Client
        
        Args:
            token: GitHub token. If None, will be read from environment
        """
        self.token = token or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.token:
            raise ValueError("GitHub Personal Access Token is required")
        
        self.server_params = StdioServerParameters(
            command="/usr/local/bin/docker",
            args=["run", "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={self.token}", "-i", "mcp/github"],
            env=None
        )
        
    async def get_session(self):
        """
        Connect to the MCP GitHub server and return a session
        
        Returns:
            A connected MCP ClientSession object
        """
        print("Connecting to GitHub MCP server...")
        
        try:
            # Open the connection to the server using async context manager
            async with stdio_client(self.server_params) as (read, write):
                self.connection = (read, write)
                
                # Create and initialize the session
                async with ClientSession(read, write) as session:
                    self.session = session
                    await self.session.initialize()
                    
                    print("Connected successfully!")
                    return self.session
                    
        except Exception as e:
            print(f"Error occurred while connecting: {e}")
            raise
    
    async def close(self):
        """Close the session and connection if they exist"""
        if hasattr(self, 'session'):
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'connection'):
            await self.connection[0].__aexit__(None, None, None)
            await self.connection[1].__aexit__(None, None, None)

# Example usage
async def example():
    client = MCPClient()
    try:
        session = await client.get_session()
        
        # List available tools
        print("Fetching available tools...")
        tools = await session.list_tools()
        print(f"Available tools: {tools}")
        
        # You can now use the session for other operations
        
    finally:
        # Ensure resources are closed
        await client.close()

if __name__ == "__main__":
    asyncio.run(example())