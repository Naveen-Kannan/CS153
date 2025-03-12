#!/usr/bin/env python3
"""
Simplified MCP GitHub Tool Pipeline - Uses Agent Planner and Pipeline Executor
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from agent_planner import AgentPlanner
from pipeline_executor import PipelineExecutor
from mcp_client import MCPClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("github-tool-pipeline")

async def run_pipeline(user_query: str):
    """
    Run the simplified pipeline with Agent Planner and Pipeline Executor.
    
    Args:
        user_query: The user's request to process
    """
    logger.info(f"Processing user query: {user_query}")
    
    # Create MCP client
    client = MCPClient()
    
    try:
        # Get MCP session
        session = await client.get_session()
        
        # Get available tools
        tools_response = await session.list_tools()
        logger.info(f"Tools available: {len(tools_response.tools)}")
        
        # Create the planner
        planner = AgentPlanner(tools_response.tools)
        
        # Generate a plan based on the user query
        logger.info("Generating plan...")
        plan = await planner.generate_plan(user_query, [])
        
        if not plan:
            logger.error("Failed to generate plan")
            return None
        
        logger.info(f"Plan generated: {json.dumps(plan, indent=2)}")
        
        # Create the executor
        executor = PipelineExecutor(plan, tools_response.tools, session)
        
        # Execute the plan
        logger.info("Executing plan...")
        result = await executor.execute_plan()
        
        logger.info("Execution complete")
        return result
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise
    finally:
        try:
            # Close client connection
            if client:
                await client.close()
        except Exception as close_error:
            logger.error(f"Error closing client: {close_error}")

async def main():
    """Main entry point"""
    
    user_query = "Get information about my GitHub repositories. My username is aliyanishfaq"
    
    if not user_query:
        user_query = input("Enter your query: ")
    
    results = await run_pipeline(user_query)
    
    if results:
        print("\n=== EXECUTION RESULTS ===")
        print(f"Summary: {results['summary']}")
        print("\nDetailed Results:")
        for idx, item in enumerate(results['context_items']):
            print(f"\nStep {idx + 1}: {item['tool_name']}")
            print(f"Input: {json.dumps(item['tool_input'], indent=2)}")
            print(f"Output: {item['tool_output']}")
            print(f"Status: {item['output_status']}")

if __name__ == "__main__":
    asyncio.run(main())