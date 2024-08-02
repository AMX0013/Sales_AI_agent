import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
# Deprecated 
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase

from prompt import create_agent_prompt

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



def load_llm():
    chat_groq = ChatGroq(temperature=0.8, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return chat_groq


def get_response_llm(user_question, memory):
    
    input_prompt = create_agent_prompt()
    chat_groq = load_llm()

    db = SQLDatabase.from_uri(os.getenv("DATABASE_URI").replace("postgres://", "postgresql://"))
    
    
    toolkit = SQLDatabaseToolkit(db=db, llm=chat_groq)
    tools = toolkit.get_tools()
    print(tools)
    
    agent_core = create_sql_agent(
        llm=chat_groq,
        db = db,
        agent_type="openai-tools",        
        verbose=True
    )

    # response = agent_core.invoke({
    #     input_prompt
    #     "question": user_question,
    #     "chat_history": memory.buffer,
    # })

    return response['text']