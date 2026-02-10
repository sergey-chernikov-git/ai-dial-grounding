import asyncio
import json
import time
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

user_client = UserClient()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small-1",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY)
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY)
)


class AIHobbySearchResponse(BaseModel):
    hobbies: dict[str, list[int]] = Field(default_factory=dict, description="Dictionary where keys are hobbies and values are lists of user IDs who mentioned those hobbies in their about_me section")


SYSTEM_PROMPT = """
You are professional assistant for searching users by their hobbies. 
You have access to the list of users with their `id` and `about_me` section, which contains information about their hobbies. 
Your task is to analyze the user's query and extract relevant hobbies mentioned in the `about_me` sections of the users.
As response please use the following format:

## Response Format:
{format_instructions}

return all ids of users that have mentioned hobby in their `about_me` section.
"""

USER_PROMPT = """
#RAG CONTEXT:
{context}

#USER QUESTION:
{query}
"""

tasks = []

def format_user(user):
    return f"User ID: {user.get('id')}, About Me: {user['about_me']}"

class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        users = user_client.get_all_users()
        start_time = time.time()
        print("Creating vectorstore with users' about_me sections...")
        self.vectorstore = Chroma(collection_name="users", embedding_function=self.embeddings)
        await self.update_vectorstore(users)
        end_time = time.time()
        print(f"Vectorstore created. {end_time - start_time} seconds")
        return self

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.1) -> str:
        docs = self.vectorstore.similarity_search_with_relevance_scores(query=query, k=k, score_threshold=score)
        context_parts = []
        for doc in docs:
            context_parts.append(doc[0].page_content)
            print(f"Score: {doc[1]}, Content: {doc[0].page_content}")
        return "\n\n".join(context_parts)

    async def update_vectorstore(self, users: list[dict[str, Any]]):
        batch_users = [user_batch for user_batch in [users[i:i + 100] for i in range(0, len(users), 100)]]
        tasks = []
        for batch in batch_users:
            tasks.append(self.vectorstore.aadd_documents(
                documents=[Document(id=user.get('id'), page_content=format_user(user)) for user in batch]
            ))
        await asyncio.gather(*tasks)


    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(query=query, context=context)

    def generate_answer(self, augmented_prompt: str) -> str:
        # messages = [
        #     SystemMessage(SYSTEM_PROMPT),
        #     HumanMessage(augmented_prompt)
        # ]
        # parser = PydanticOutputParser(pydantic_object=AIHobbySearchResponse)
        # prompt = ChatPromptTemplate.from_messages(messages)
        # chain = self.llm_client | prompt
        # response = self.llm_client.invoke(messages)

        messages = [
            SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        parser = PydanticOutputParser(pydantic_object=AIHobbySearchResponse)
        prompt = ChatPromptTemplate.from_messages(messages).partial(
            format_instructions=parser.get_format_instructions())
        response = (prompt | self.llm_client | parser).invoke({})
        return response.hobbies

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

async def main():
    # task = asyncio.create_task(user_retrieval())
    while True:

        print("== Hobbies Searching Wizard ==")
        print("Hello and welcome to Hobbies Searching Wizard! Please enter your request:")

        user_input = input("> ")
        hobbies = []
        async with UserRAG(embeddings, llm) as user_rag:
            context = await user_rag.retrieve_context(user_input)
            augmented_prompt = user_rag.augment_prompt(user_input, context)
            hobbies = user_rag.generate_answer(augmented_prompt)

        for hobby in hobbies:
            print(f"Hobby: {hobby}")
            print("Users:")
            for user_id in hobbies[hobby]:
                user_info = await user_client.get_user(int(user_id))
                print(json.dumps(user_info, indent=2))


asyncio.run(main())

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)
