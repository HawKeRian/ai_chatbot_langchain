import chainlit as cl
from agent import agent_graph  # Import your backend module

@cl.on_chat_start
async def start():
    # Set thread_id for memory isolation [cite: 241]
    cl.user_session.set("config", {"configurable": {"thread_id": "session_1"}})
    
    
@cl.on_message
async def main(message: cl.Message):
    config = cl.user_session.get("config")
    
    # Run the graph
    response = agent_graph.invoke(
        {"messages": [("user", message.content)]},  # type: ignore
        config=config
    )
    
    # Send result
    await cl.Message(content=response["messages"][-1].content).send()