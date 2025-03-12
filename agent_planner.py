#!/usr/bin/env python3
"""
Agent Planner - Generates execution plans for tools based on user requests
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import os

from openai import OpenAI
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp_client import MCPClient

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent-planner")

class AgentPlanner:
    """Plans a sequence of tool calls based on user requests"""
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the Agent Planner.
        
        Args:
            tools: List of Tool objects with name, description, and inputSchema
        """
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.openai_model = "gpt-4"  # Fixed model name typo
        self.tools = tools
    
    async def generate_plan(self, user_message: str, history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Generate an execution plan based on the user's message.
        
        Args:
            user_message: The user's request
            history: Conversation history
            
        Returns:
            A plan with steps to execute or None if planning failed
        """
        # Format tools information for the prompt
        tools_info = []
        
        for tool in self.tools:
            # Extract tool details from the Tool object
            tool_name = tool.name
            tool_description = tool.description
            tool_schema = tool.inputSchema
            
            # Extract required properties from the schema
            required_props = tool_schema.get("required", [])
            
            # Format properties with required/optional labels
            properties = []
            for prop_name, prop_details in tool_schema.get("properties", {}).items():
                is_required = prop_name in required_props
                properties.append(f"    - {prop_name}: {prop_details.get('description', '')} "
                                 f"({'REQUIRED' if is_required else 'OPTIONAL'})")
            
            properties_text = "\n".join(properties)
            
            tool_info = f"""Tool: {tool_name}
            Description: {tool_schema.get('description', '')}
            Parameters:
            {properties_text}
            """
            tools_info.append(tool_info)
        
        tools_info_text = "\n\n".join(tools_info)
        
        # Create system prompt
        system_prompt = f"""
        You are a personal assistant that helps users communicate with their contacts/recipients easier
        based on the request. Your specific role is that of the planning agent. You are provided with a list of tools.
        Your job is to plan a list of steps to help achieve the user's goal. Below are a few examples for your reference:

        AVAILABLE TOOLS:
        {tools_info_text}
        
        For each step in your plan, specify:
        1. The tool to use
        2. The input parameters
        3. A variable name to store the output
        
        Your steps should be in a logical sequence to accomplish the user's request effectively.
        The following are the examples of how the response should look like:
        Example 1:
        INPUT: Send John a message reminding him about our meeting at 3 PM.
        OUTPUT:
        {{"steps": [
          {{ "tool": "contacts", "input": {{ "name": "John" }}, "output_var": "john_contact" }},
          {{ "tool": "messages", "input": {{ "operation": "send", "phoneNumber": "{{{{john_contact}}}}", "message": "Reminder: Our meeting is at 3 PM." }}}}
        ]}}

        Example 2:
        INPUT: Find a good pizza place near Fi-di and send the details to Mark.
        OUTPUT:
        {{"steps": [
          {{ "tool": "webSearch", "input": {{ "query": "Best pizza places near Fi-di" }}, "output_var": "pizza_places" }},
          {{ "tool": "contacts", "input": {{ "name": "Mark" }}, "output_var": "mark_contact" }}, 
          {{ "tool": "messages", "input": {{ "operation": "send", "phoneNumber": "{{{{mark_contact}}}}", "message": "Hey Mark, here are some great pizza places near me: {{{{pizza_places}}}}" }} }}
        ]}}

        Example 3:
        INPUT: Remind me to submit my report by 5 PM today.
        OUTPUT:
        {{"steps": [
          {{ "tool": "reminders", "input": {{ "operation": "create", "name": "Submit report", "dueDate": "today 17:00" }} }}
        ]}}
        """
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history
        for msg in history:
            messages.append(msg)
        
        # Add user's current message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Make API call with retries
            for attempt in range(3):
                try:
                    response = await asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model=self.openai_model,
                        messages=messages,
                        response_format={"type": "json_object"}
                    )
                    
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("No content found in the response")
                    
                    # Parse and validate response
                    parsed = json.loads(content)
                    if "steps" not in parsed or not isinstance(parsed["steps"], list):
                        raise ValueError("Response doesn't contain a valid steps array")
                    
                    # Validate tools exist
                    tool_names = [tool.get("name") for tool in self.tools]
                    for step in parsed["steps"]:
                        if step.get("tool") not in tool_names:
                            raise ValueError(f"Tool {step.get('tool')} not found")
                    
                    return parsed
                
                except Exception as e:
                    logger.error(f"[AgentPlanner] Retry attempt {attempt+1}/3 failed: {str(e)}")
                    if attempt == 2:  # Last attempt
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            logger.error(f"[AgentPlanner] Error generating plan: {str(e)}")
            return None

# Example usage
async def example():
    # Example tools
    client = MCPClient()
    try:
        async with stdio_client(client.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print('TOOLS', tools)
                planner = AgentPlanner(tools)
                result = await planner.generate_plan(
                    "Send Alex a message about our meeting tomorrow",
                    []  # Empty history
                )
                print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(example())