import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import Annotated
from typing_extensions import TypedDict
import uuid


OPENAI_API_KEY = 'your-openai-api-key-here'

st.set_page_config(page_title="Multi-Session Chatbot", layout="wide")


# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )


# Build the graph
@st.cache_resource
def build_graph():
    llm = get_llm()
    
    # Chat node - takes state, returns updated state
    def chat_node(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    # Build graph
    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Get compiled graph
app = build_graph()


# Initialize session state for UI
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = {}  

if "current_thread" not in st.session_state:
    st.session_state.current_thread = None


# Create new thread
def create_thread():
    thread_id = str(uuid.uuid4())[:8]
    st.session_state.thread_ids[thread_id] = {
        "title": "New Chat",
        "display_history": []
    }
    return thread_id


# Generate title from first message
def generate_title(message):
    title = message[:30].strip()
    if len(message) > 30:
        title += "..."
    return title


# Sidebar for thread management
with st.sidebar:
    st.title("Chat Sessions")
    
    # New chat button
    if st.button("+ New Chat", use_container_width=True):
        new_id = create_thread()
        st.session_state.current_thread = new_id
        st.rerun()
    
    st.divider()
    
    # List existing threads
    if st.session_state.thread_ids:
        for thread_id, thread_data in st.session_state.thread_ids.items():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if thread_id == st.session_state.current_thread:
                    if st.button(f"▶ {thread_data['title']}", key=f"select_{thread_id}", use_container_width=True):
                        pass
                else:
                    if st.button(thread_data['title'], key=f"select_{thread_id}", use_container_width=True):
                        st.session_state.current_thread = thread_id
                        st.rerun()
            
            with col2:
                if st.button("🗑", key=f"delete_{thread_id}"):
                    del st.session_state.thread_ids[thread_id]
                    if st.session_state.current_thread == thread_id:
                        st.session_state.current_thread = None
                    st.rerun()
    else:
        st.caption("No chats yet. Start a new one")
    



# Main chat area
st.title("Multi-Session Chatbot")
st.caption("A conversational agent with memory that maintains context across turns")

# If no thread selected, prompt to create one
if st.session_state.current_thread is None:
    st.info("Start a new chat from the sidebar to begin.")
else:
    current = st.session_state.thread_ids[st.session_state.current_thread]
    thread_id = st.session_state.current_thread
    
    # Config for this thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Display chat history
    for msg in current["display_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add to display history
        current["display_history"].append({"role": "user", "content": user_input})
        
        # Update title if first message
        if len(current["display_history"]) == 1:
            current["title"] = generate_title(user_input)
        
        # Invoke graph with thread config
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = app.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )
                response = result["messages"][-1].content
            st.write(response)
        
        # Add to display history
        current["display_history"].append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Show checkpoint info in expander
    with st.expander("Checkpoint"):
        try:
            state = app.get_state(config)
            st.write("**Stored Messages in Checkpoint:**")
            for msg in state.values.get("messages", []):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                st.write(f"- **{role}:** {msg.content[:100]}...")
        except Exception as e:
            st.caption(f"No checkpoint yet or error: {e}")
