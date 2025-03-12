import os
import discord
from openai import AsyncOpenAI

# Models
GPT4_MINI = "gpt-4o-mini"
O1_MODEL = "o1"

# System prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
CODE_SYSTEM_PROMPT = "You are a helpful coding assistant. Provide code solutions in a clear, well-commented format. Always wrap your code in appropriate markdown code blocks with the language specified."

class OpenAIAgent:
    def __init__(self):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: discord.Message):
        # Check if this is a code request
        is_code_request = message.content.lower().startswith("code:")
        
        # Select the appropriate model and system prompt
        if is_code_request:
            model = O1_MODEL
            system_prompt = CODE_SYSTEM_PROMPT
            # Remove the "code:" prefix from the message
            user_message = message.content[5:].strip()
        else:
            model = GPT4_MINI
            system_prompt = DEFAULT_SYSTEM_PROMPT
            user_message = message.content
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Call the OpenAI API
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content
