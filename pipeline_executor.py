#!/usr/bin/env python3
"""
Pipeline Executor - Executes tool pipelines based on execution plans
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline-executor")

class OutputStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class PipelineExecutor:
    """Executes tool pipelines based on execution plans"""
    
    def __init__(self, plan: Dict[str, Any], tools: List[Any], mcp_service: Any):
        """
        Initialize the Pipeline Executor.
        
        Args:
            plan: Execution plan with steps
            tools: List of available tools
            mcp_service: MCP service instance for tool execution
        """
        self.context = {"instruction": "", "context_items": []}
        self.plan = plan
        self.tools = tools
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.openai_model = "gpt-4"
        self.mcp_service = mcp_service
        
    def clean_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean schema to be compatible with OpenAI's API"""
        if not schema or not isinstance(schema, dict):
            return schema
            
        cleaned = schema.copy()
        
        if "properties" in cleaned:
            for prop_name, prop_details in cleaned["properties"].items():
                if prop_details and isinstance(prop_details, dict):
                    # Collect unsupported properties
                    unsupported_props = []
                    
                    if "default" in prop_details:
                        unsupported_props.append(f"Default: {prop_details['default']}")
                        del prop_details["default"]
                    
                    # Add unsupported properties to description
                    if unsupported_props:
                        prop_details["description"] = prop_details.get("description", "")
                        prop_details["description"] += f" ({', '.join(unsupported_props)})"
                        
        return cleaned

    async def get_openai_response(self, messages: List[Dict[str, str]], model: str = None, max_retries: int = 3) -> Optional[str]:
        """Get a simple text response from OpenAI"""
        model = model or self.openai_model
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=model,
                    messages=messages
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response received from OpenAI")
                    
                return content
                
            except Exception as e:
                logger.error(f"[PipelineExecutor] Text response retry attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        return None

    async def get_openai_structured_response(self, messages: List[Dict[str, str]], schema: Dict[str, Any], schema_name: str = "response", model: str = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Get a structured response from OpenAI based on provided schema"""
        model = model or self.openai_model
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_object",
                        "schema": schema
                    }
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty structured response received from OpenAI")
                    
                return json.loads(content)
                
            except Exception as e:
                logger.error(f"[PipelineExecutor] Structured response retry attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        return None

    async def create_parameters(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameters for a tool execution step"""
        # Find the tool
        tool = next((t for t in self.tools if t.name == step["tool"]), None)
        if not tool:
            logger.error(f"[PipelineExecutor] Tool {step['tool']} not found")
            return {}

        # Format context items
        context_section = ""
        if self.context["context_items"]:
            context_items = []
            for i, item in enumerate(self.context["context_items"]):
                context_items.append(f"""
                Context Item {i + 1}:
                - Tool Used: {item['tool_plan']}
                - Input: {json.dumps(item['tool_input'])}
                - Output: {json.dumps(item['tool_output'])}
                - Status: {item['output_status']}
                """)
            context_section = "The conversation context is as follows:\n" + "\n".join(context_items)

        prompt = f"""
        You are an AI assistant that is part of an agent pipeline that serves as a personal assistant to the user.
        Your specific role is to create the parameters for the tool "{step['tool']}".
        
        Tool details:
        - Parameters schema: {json.dumps(tool.inputSchema.get('properties', {}), indent=2)}
        - Required parameters: {json.dumps(tool.inputSchema.get('required', []))}
        
        Please create the parameters for this tool in a valid JSON format.
        {context_section}

        User provided the following rough input for the tool:
        {json.dumps(step['input'])}
        
        Based on this input and schema requirements, provide a properly formatted parameter object for the tool.
        """

        logger.info(f"[PipelineExecutor][create_parameters] Prompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        
        tool_input_schema = self.clean_schema_for_openai({
            **(tool.inputSchema or {}),
            "additionalProperties": False,
            "required": list(tool.inputSchema.get('properties', {}).keys()) if tool.inputSchema else []
        })
        
        try:
            response = await self.get_openai_structured_response(messages, tool_input_schema, "parameters")
            return response or {}
        except Exception as e:
            logger.error(f"[PipelineExecutor][create_parameters] Failed to create parameters: {str(e)}")
            return {}

    async def refine_tool_output(self, context_item: Dict[str, Any]) -> Optional[str]:
        """Refine the output of a tool execution"""
        prompt = f"""
        You are an AI assistant that is part of an agent pipeline that serves as a personal assistant to the user.
        Your specific role is to refine the output of the tool that was invoked so we can use more concise information
        in downstream usage.

        The following was the plan for the tool: {context_item['tool_plan']}
        The following was the input to the tool: {json.dumps(context_item['tool_input'])}
        The following was the output of the tool: {json.dumps(context_item['tool_output'])}

        Please refine the output of the tool in the form of a string that provides the most concise information. Don't compromise
        on details, but do your best to provide a concise result. Don't add asterisk or complex formatting symbols.
        """

        messages = [{"role": "user", "content": prompt}]
        return await self.get_openai_response(messages)

    async def plan_execution_summary(self) -> Optional[str]:
        """Generate a summary of the execution plan results"""
        context_items = ""
        if self.context["context_items"]:
            items = []
            for i, item in enumerate(self.context["context_items"]):
                items.append(f"""
                Step {i + 1}: {item['tool_name']}
                - Description: {item['tool_plan']}
                - Input Parameters: {json.dumps(item['tool_input'], indent=2)}
                - Output Result: {item['tool_output']}
                - Status: {item['output_status']}
                """)
            context_items = "\n".join(items)

        prompt = f"""
        You are an AI assistant that helps users by summarizing the results of executed tool pipelines.
        
        EXECUTION PLAN:
        {json.dumps(self.plan, indent=2)}
        
        EXECUTION RESULTS:
        {context_items or "No tools were executed."}
        
        Please provide a clear, concise summary of what was done and what was found so that it can be stored in records.
        We want to know the execution plan and what was provided to the user at the end of the execution. I don't want fancy formatting.
        Please provide the summary in a readable format and cap it at 5 lines (max 10 lines for execution longer than 3 steps).

        There is no need to mention how many tools were executed or specifics of the tool names. Your name is Eva. Don't try to use technical lingo/jargon since
        the message will be displayed on a user friendly UI.
        """
        
        logger.info(f"[PipelineExecutor][plan_execution_summary] Prompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        return await self.get_openai_response(messages)

    async def plan_display_text(self, tool_input: Any) -> Optional[str]:
        """Generate a display text for a tool execution"""
        prompt = f"""
        You are an AI model that is part of a larger personal AI assistant system.
        Your task is to generate a display text of 2-3 words based on the input that is being
        passed to the tool call. The display text should be user friendly and concise.

        The display text should only be words, no asterisks or formatting symbols, or emojis.
        {json.dumps(tool_input, indent=2)}
        """
        
        messages = [{"role": "user", "content": prompt}]
        return await self.get_openai_response(messages)

    async def execute_step(self, step: Dict[str, Any]) -> None:
        """Execute a single step of the plan"""
        context_item = {
            "tool_name": step["tool"],
            "tool_plan": f"tool name: {step['tool']} tool input: {step['input']}",
            "tool_input": {},
            "tool_output": {},
            "output_status": OutputStatus.FAILURE
        }

        parameters = await self.create_parameters(step) or {}
        context_item["tool_input"] = parameters

        tool_result = await self.mcp_service.process_tool_use(step["tool"], parameters)
        if not tool_result:
            raise ValueError("[PipelineExecutor][execute_step] Tool execution failed")
        
        context_item["tool_output"] = tool_result
        refined_output = await self.refine_tool_output(context_item)
        if refined_output:
            context_item["tool_output"] = refined_output
        context_item["output_status"] = OutputStatus.SUCCESS

        self.context["context_items"].append(context_item)

    async def execute_plan(self) -> Dict[str, Any]:
        """Execute the entire plan"""
        logger.info(f"Executing plan: {self.plan}")
        
        for step in self.plan["steps"]:
            logger.info(f"Executing step: {step}")
            display_text = await self.plan_display_text(step["input"])
            await self.execute_step(step)
            
        summary = await self.plan_execution_summary()
        logger.info(f"[PipelineExecutor][execute_plan] Execution summary: {summary}")
        
        return {
            "summary": summary,
            "context_items": self.context["context_items"]
        }