import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


llm_client = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_key=SecretStr(API_KEY),
    azure_endpoint=DIAL_URL,
    api_version=""
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    res = ""
    for user in context:
        res += f"User:\n"
        for key, value in user.items():
            if isinstance(value, dict):
                res += f"  {key}:\n"
                for sub_key, sub_value in value.items():
                    res += f"    {sub_key}: {sub_value}\n"
            else:
                res += f"  {key}: {value}\n"
        res += "\n"
    return res


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_message
        )
    ]
    res = await llm_client.ainvoke(messages)
    total_tokens = res.usage_metadata['total_tokens']
    content = res.content
    token_tracker.add_tokens(total_tokens)
    print(f"LLM Response content: {content}")
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")
    await generate_response("Me is system", "Give me an random advice")
    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")
        users = UserClient().get_all_users()
        users_batch = [users[i:i+100] for i in range(0, len(users), 100)]
        tasks = [
            asyncio.create_task(
                generate_response(
                    BATCH_SYSTEM_PROMPT,
                    USER_PROMPT.format(context=users, query=user_question)
                )
            )
            for users in users_batch]
        res = await asyncio.gather(*tasks)
        out = [r for r in res if r != "NO_MATCHES_FOUND"]
        if not out:
            print("User not found matching criteria")
        else:
            res = await generate_response(FINAL_SYSTEM_PROMPT, USER_PROMPT.format(context="\n\n".join(out), query=user_question))
            print(res)
        print(f"Token used: {token_tracker.get_summary()}")


if __name__ == "__main__":
    asyncio.run(main())
