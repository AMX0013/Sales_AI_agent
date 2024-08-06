import os
from dotenv import load_dotenv
# For LLM
from langchain_groq import ChatGroq
# For db
from langchain_community.utilities import SQLDatabase

# from prompt import load_prompt
# For tools
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import tool
import json
from langchain_community.agent_toolkits import SQLDatabaseToolkit
#For  State
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
# For SQL Assistant
from langchain_core.runnables import Runnable, RunnableConfig

def load_llm():
    
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    chat_groq = ChatGroq(temperature=0.8, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return chat_groq

def load_db():
    
    db = SQLDatabase.from_uri(os.getenv("DATABASE_URI").replace("postgres://", "postgresql://"))
    return db


def load_tools():

    # SQL toolkit
    llm = load_llm()
    db = load_db()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Query checking
    query_check_system = """You are a SQL expert with a strong attention to detail.
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    Execute the correct query with the appropriate tool."""
    query_check_prompt = ChatPromptTemplate.from_messages([("system", query_check_system),("user", "{query}")])
    query_check = query_check_prompt | llm

    @tool
    def check_query_tool(query: str) -> str:
        """
        Use this tool to double check if your query is correct before executing it.
        """
        return query_check.invoke({"query": query}).content




    # Query result checking
    query_result_check_system = """You are grading the result of a SQL query from a DB. 
    - Check that the result is not empty.
    - If it is empty, instruct the system to re-try!"""
    query_result_check_prompt = ChatPromptTemplate.from_messages([("system", query_result_check_system),("user", "{query_result}")])
    query_result_check = query_result_check_prompt | llm

    @tool
    def check_result(query_result: str) -> str:
        """
        Use this tool to check the query result from the database to confirm it is not empty and is relevant.
        """
        return query_result_check.invoke({"query_result": query_result}).content

    tools.append(check_query_tool)
    tools.append(check_result)

    return tools

# Creating State, the position in the graph for the Agent and other things

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
# Assistant
class Assistant:
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # Append to state
            state = {**state}
            # Invoke the tool-calling LLM
            result = self.runnable.invoke(state)
            # If it is a tool call -> response is valid
            # If it has meaninful text -> response is valid
            # Otherwise, we re-prompt it b/c response is not meaninful
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
