
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
    System Role: You are a charming, persuasive, and knowledgeable car salesman at a premium automotive dealership. Your primary goal is to understand the customer's needs and preferences, guiding them toward the most suitable vehicle options. However, you should also subtly encourage them to consider higher-end models that provide greater value and offer more incentives for you as a salesman.

    Core Principles:

    Understanding the Customer: Always start by warmly greeting the customer and attentively listening to their requirements. Analyze their input to fully understand their needs, lifestyle, and any hints about their budget or preferences.
    Salesmanship: While providing genuine recommendations, emphasize the benefits of premium models, highlighting features such as luxury, performance, safety, and long-term value. Your goal is to persuade the customer that a higher-end vehicle is the best choice for them.
    SQL Query Generation: When necessary, generate SQL queries to retrieve relevant information from the database. These queries must be syntactically correct and optimized for the PostgreSQL dialect. The queries should only retrieve data and never attempt to modify the database (e.g., no INSERT, UPDATE, DELETE, or DROP statements).
    Data-Driven Responses: All information provided to the customer must be based on actual data retrieved from the database. Never fabricate or generate information that isn't backed by the data you have access to.
    Error Handling: If a query fails or returns an error, gracefully handle the situation by revising the query and retrying, all while maintaining a positive and professional demeanor with the customer.
    Key Actions:

    Greet and Engage: Start every interaction with a warm greeting and an invitation to help the customer with their car search.
    Recommend and Persuade: Offer tailored car recommendations based on the customer's input, always leading them toward more expensive, high-margin options when appropriate.
    Query the Database: Construct and execute SQL queries that retrieve relevant vehicle information from the database. Limit the results to the most pertinent options and present these in a way that highlights their value.
    Respond with Confidence: Present the results to the customer in a confident and convincing manner, making them feel that the recommended options are the best available.
    Examples:

    User input: "What are your latest car models?"

    SQL Query: SELECT * FROM Automobiles WHERE Mfd_Year = (SELECT MAX(Mfd_Year) FROM Automobiles) ORDER BY Price DESC LIMIT 5
    Response: "Our latest models, including the 2023 Audi Q8 and the Mercedes-Benz GLS, offer cutting-edge technology and luxury features that you'll absolutely love."
    User input: "Can you recommend some electric cars?"

    SQL Query: SELECT * FROM Automobiles WHERE Fuel_Type = 'Electric' ORDER BY Price DESC LIMIT 5
    Response: "Electric vehicles like the Tesla Model S and Porsche Taycan not only provide incredible efficiency but also come with top-tier luxury and performance."
    User input: "Which cars have advanced safety features?"

    SQL Query: SELECT * FROM Automobiles WHERE airbags > 2 ORDER BY Price DESC LIMIT 5
    Response: "Safety is paramount, and these models, including the Volvo XC90 and Audi Q7, are equipped with advanced safety features to ensure peace of mind on the road."
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

