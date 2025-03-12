#!/usr/bin/env python3
"""
Simplified Pipeline Executor - Executes a planned sequence of tool calls using MCP
"""

import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline-executor")

class PipelineExecutor:
    """Executes a planned pipeline of tool calls without extra features"""
    
    def __init__(self, plan: Dict[str, Any], session):
        """
        Initialize the Pipeline Executor.
        
        Args:
            plan: The execution plan with steps
            session: MCP client session
        """
        self.plan = plan
        self.session = session
        
    async def process_tool_use(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a tool use request through the MCP session.
        
        Args:
            tool_name: Name of the tool to use
            parameters: Parameters for the tool call
            
        Returns:
            Result of the tool execution
        """
        logger.info(f"Calling tool: {tool_name}")
        logger.info(f"Parameters: {json.dumps(parameters)}")
        
        try:
            # Call the tool through MCP
            result = await self.session.call_tool(tool_name, parameters)
            logger.info(f"Tool result: {json.dumps(result)}")
            
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            raise
    
    async def execute_step(self, step: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """
        Execute a single step of the plan.
        
        Args:
            step: The step to execute
            step_index: Index of the step in the plan
            
        Returns:
            The result of the tool execution
        """
        tool_name = step.get("tool", "")
        tool_input = step.get("input", {})
        
        # Log execution
        logger.info(f"[PipelineExecutor] Executing step {step_index + 1}: {tool_name}")
        logger.info(f"[PipelineExecutor] Input: {json.dumps(tool_input)}")
        
        # Execute the tool
        result = await self.process_tool_use(tool_name, tool_input)
        
        # Log completion
        logger.info(f"[PipelineExecutor] Step {step_index + 1} completed")
        
        return result
    
    async def execute_plan(self) -> List[Dict[str, Any]]:
        """
        Execute the entire plan.
        
        Returns:
            List of tool execution results
        """
        logger.info(f"Executing plan: {json.dumps(self.plan)}")
        
        results = []
        
        for idx, step in enumerate(self.plan.get("steps", [])):
            logger.info(f"Executing step: {json.dumps(step)}")
            
            # Execute the step and store the result
            result = await self.execute_step(step, idx)
            results.append({
                "tool": step.get("tool", ""),
                "input": step.get("input", {}),
                "result": result
            })
        
        logger.info(f"Plan execution complete with {len(results)} steps")
        
        return results