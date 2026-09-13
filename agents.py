from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
import os

# @tool
# def search_internet(query: str) -> str:
#     """
#     Search the internet for information related to the provided query.
#
#     This tool uses TavilySearch to perform web searches and retrieve relevant
#     information that can be used to answer queries or provide context.
#
#     :param query: The search query string to look up on the internet
#     :return: A string containing the search results and relevant information
#     """
#     print(f'input query {query}')
#     search = TavilySearch()
#     result = search.run(query)
#     return result

if __name__ == '__main__':
    load_dotenv()
    tools = [TavilySearch()]
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_HOST"))
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": [HumanMessage(content="what is the result of fiba world cup this summer")]})
    print(response["messages"][-1].content)
