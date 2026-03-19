# Conversational Chatbot

A chatbot built with LangGraph that maintains conversation context across multiple turns. It features a Streamlit web interface with multiple chat sessions, similar to ChatGPT's sidebar navigation.


## Requirements

Install the dependencies:

```
pip install streamlit langchain langchain-openai langgraph 
```


## API Key Setup

This application requires an OpenAI API key.

1. Create a `.env` file in the same folder as `chatbot.py`
2. Add your API key to the file:

```
OPENAI_API_KEY=your-openai-api-key-here
```

To get an API key, go to https://platform.openai.com/api-keys, sign-in to your account and create a new secret key.


## How to Run

Run the application from the command line:

```
streamlit run chatbot.py
```

The app will open a local host in your browser.


## How to Run

1. Click "+ New Chat" in the sidebar to start a conversation.
2. Type messages in the chat input.
3. The chatbot remembers everything you've discussed in that session
4. Create multiple sessions to have separate conversations.
5. Click the trash icon to delete a session.
6. Expand "Checkpoint" to see what's stored in the checkpoint.


## How It Works

The chatbot uses LangGraph's checkpointing system to maintain conversation state.

1. **User Input** - The user enters a message in the Streamlit chat input
2. **Thread Selection** - Each chat session has a unique thread_id that identifies it
3. **State Loading** - LangGraph loads the existing message history for that thread from the MemorySaver checkpointer
4. **LLM Invocation** - The full conversation history plus the new message is sent to GPT-4o-mini
5. **Checkpointing** - After the response, the updated state is saved back to the checkpointer
6. **Output Display** - The response is displayed in the chat interface and added to the session history
