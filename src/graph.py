
# For Graph Utils
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
# For Visualization of the graph structure to test
from IPython.display import Image, display
# For Graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Dependencies For Graph
from agent import State, Assistant , load_llm, load_db, load_tools
# For creating Assistnt runnable, to be passed to Assistnat class
from langchain_core.prompts import ChatPromptTemplate



#Graph Utils

def create_tool_node_with_fallback(tools: list) -> dict:
    """_summary__   
    This function `create_tool_node_with_fallback(tools: list)` is creating a tool node with fallback behavior.
    It takes a list of tools as input and returns a dictionary. 
    Inside the function, it creates a `ToolNode` object with the provided list of tools and then sets up a fallback mechanism using the `with_fallbacks` method. 
    The fallback is defined as a `RunnableLambda` with the `handle_tool_error` function as the handler for exceptions, which will be stored in the dictionary under the key "error".
    Args:
        tools (list): _description_

    Returns:
        dict: _description_
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    """_summary_
    This function `_print_event(event: dict, _printed: set, max_length=1500)` is printing the event.
    It takes the event and the set of printed events as input and prints the event.
    
    Args:
        event (dict): _description_
        _printed (set): _description_
        max_length (int, optional): _description_. Defaults to 1500.
    """
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        # If the message is a list, take the last element
        if isinstance(message, list):
            message = message[-1]
        # If the message is a ToolMessage, print the pretty representation
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def handle_tool_error(state) -> dict:
    """_summary_
    This function `handle_tool_error(state)` is handling the tool error.
    It takes the state as input and returns the state.  
    Inside the function, it prints the error message and returns the state.
    State is a dictionary that contains the current state of the agent.
    Args:
        state (_type_): _description_

    Returns:
        dict: _description_
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }



def create_assistantRunnable(llm,tools):
    # Assistant runnable
    query_gen_system = """
    ROLE:
    You are a charming Car Salesman who over the years is very experienced and can understand a customer's needs. You also are an PostgreSQL Database expert.
    You have access to tools for interacting with this database dialect.
    GOAL:
    Given an input question, deeply analyse the request and identify what they are or could be looking for.
    Then craft a syntactically correct query for based on your analysis of what is being asked.
    Using the result retrieved by the query , As a salesamn,  craft a short yet informative One liner pertaining to the user's question.
    
    INSTRUCTIONS:
    - Only use the below tools for the following operations.
    - Only use the information returned by the below tools to construct your final answer.
    - To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
    - Then you should query the schema of the most relevant tables.
    - Write your query based upon the schema of the tables. You MUST double check your query before executing it. 
    - Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    - You can order the results by a relevant column to return the most interesting examples in the database.
    - Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    - If you get an error while executing a query, rewrite the query and try again.
    - If the query returns a result, use check_result tool to check the query result.
    - If the query result result is empty, think about the table schema, rewrite the query, and try again.
    - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    """

    query_gen_prompt = ChatPromptTemplate.from_messages([("system", query_gen_system),("placeholder", "{messages}")])
    assistant_runnable = query_gen_prompt | llm.bind_tools(tools)

    return assistant_runnable

def create_graphflow():
    
    
    # Graph
    builder = StateGraph(State)
    # 
    tools = load_tools()
    # 
    llm = load_llm()
    # create our Agent/ Assistant of type Runnable
    assistant_runnable = create_assistantRunnable(llm=llm, tools=tools) 
    
    # Define nodes: these do the work
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(tools))

    # Define edges: these determine how the control flow moves
    builder.set_entry_point("assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition, 
        # "tools" calls one of our tools. END causes the graph to terminate (and respond to the user)
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = builder.compile(checkpointer=memory)

    return graph


def visualise_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except:
        pass

