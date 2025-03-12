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
        self.openai_model = "gpt-4o"  # Fixed model name typo
        self.tools = tools
    
    def _format_tool_info(self, tool: Any) -> str:
        """Format information about a single tool
        
        Args:
            tool: Tool object containing name, description, and schema
            
        Returns:
            str: Formatted tool information including name, description, and parameters
        """
        # Extract tool details
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
        
        return f"""Tool: {tool_name}
        Description: {tool_description}
        Parameters:
        {properties_text}
        """
    
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
        
        # Format information for each tool
        tools_info = [self._format_tool_info(tool) for tool in self.tools]
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

        THE RESPONSE MUST BE IN A JSON
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
                    tool_names = [tool.name for tool in self.tools]
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

    async def generate_params(self, user_message, tools, step, context):

            # Format tools information for the prompt
        tools_info = []
        
        # Format information for each tool
        tools_info = [self._format_tool_info(tool) for tool in self.tools]
        tools_info_text = "\n\n".join(tools_info)
        prompt = """
        You are an AI github assistant. Specifically, you are given a step from a plan.
        You need to generate the input parameters for the tool call.

        CONTEXT FROM PREVIOUS STEPS:
        {context}

        Here is the step:
        {step}

        Here are the tools available to you:
        {tools_info_text}

        Here is the user's message:
        {user_message}

        The response should be in JSON
        """
        formatted_prompt = prompt.format(
            step=step,
            tools_info_text=tools_info_text,
            user_message=user_message,
            context=context
        )
        print('FORMATTED PROMPT', formatted_prompt)
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        print('CONTENT', content)
        return content

    async def tool_call_summary(self, result):
        prompt = """
        You are an AI github assistant. Specifically, you are given a result from a tool call.
        You need to summarize the results properly, making sure all the details are included.

        The formatting needs to be very simple

        Here is the result:
        {result}
        """
        formatted_prompt = prompt.format(
            result=result
        )
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        content = response.choices[0].message.content
        return content

# Changed function to take input from terminal
async def main():
    # Get user input from terminal
    user_message = input("What would you like me to help you with? ")
    
    # Example tools
    client = MCPClient()
    try:
        # Validate GitHub token before proceeding
        github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not github_token:
            print("ERROR: No GitHub token found in environment variables.")
            print("Please set GITHUB_PERSONAL_ACCESS_TOKEN in your .env file.")
            return
            
        print("Connecting to GitHub API...")
        async with stdio_client(client.server_params) as (read, write):
            try:
                context = []
                async with ClientSession(read, write) as session:
                    try:
                        await session.initialize()
                        print("Connection established successfully.")
                        
                        # Fetch available tools
                        try:
                            print("Fetching available GitHub tools...")
                            tools_response = await session.list_tools()
                            tools = tools_response.tools
                            if not tools:
                                print("WARNING: No tools available from GitHub API.")
                                return
                                
                            print(f"Successfully retrieved {len(tools)} tools.")
                            
                            # Create planner with available tools
                            planner = AgentPlanner(tools)
                            
                            # Generate plan
                            print(f"\nPlanning steps for: '{user_message}'")
                            try:
                                result = await planner.generate_plan(
                                    user_message,
                                    []  # Empty history
                                )
                                
                                if not result:
                                    print("Unable to generate a plan for your request.")
                                    return
                                    
                                print(f"Executing plan with {len(result['steps'])} steps:")
                                
                                # Execute each step with error handling
                                for i, step in enumerate(result['steps']):
                                    try:
                                        print(f"\nStep {i+1}: Using tool '{step['tool']}'")
                                        params = await planner.generate_params(
                                            user_message,
                                            tools,
                                            step,
                                            context
                                        )
                                        params = json.loads(params)
                                        print(f"  Parameters: {json.dumps(params, indent=2)}")
                                        
                                        # Execute the tool call
                                        result = await session.call_tool(step['tool'], params)
                                        summary = await planner.tool_call_summary(result)
                                        print(f"  Result: {summary}")
                                        context.append(summary)
                                    except Exception as e:
                                        logger.error(f"[AgentPlanner] Error executing step {i+1}: {str(e)}")
                                        print(f"  Error: {str(e)}")
                            except Exception as e:
                                logger.error(f"[AgentPlanner] Error generating plan: {str(e)}")
                                print("Unable to generate a plan for your request.")
                                return
                        except Exception as e:
                            logger.error(f"[AgentPlanner] Error fetching tools: {str(e)}")
                            print("Unable to fetch tools from GitHub API.")
                            return
                    except Exception as e:
                        logger.error(f"[AgentPlanner] Error initializing session: {str(e)}")
                        print("Unable to connect to GitHub API.")
                        return
            except Exception as e:
                logger.error(f"[AgentPlanner] Error connecting to GitHub API: {str(e)}")
                print("Unable to connect to GitHub API.")
                return
    except Exception as e:
        logger.error(f"[AgentPlanner] Error in main function: {str(e)}")
        print("An error occurred. Please try again later.")

if __name__ == "__main__":
    asyncio.run(main())