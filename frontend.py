import streamlit as st
from langchain_app import initialize_bot, set_api_key
from streamlit_local_storage import LocalStorage
import logging

localS = LocalStorage()

# Initialize session state
if 'api_key_entered' not in st.session_state:
    st.session_state.api_key_entered = False

# Load API key from local storage if available
stored_api_key = localS.getItem("google_api_key")
if stored_api_key:
    set_api_key(stored_api_key)
    st.session_state.api_key_entered = True
    st.session_state.GOOGLE_API_KEY = stored_api_key

st.title("Mr.G (IIITK AI Assistant)")

# API Key Management - Must be at the top
if not st.session_state.api_key_entered:
    st.write("Welcome! Please enter your Google API key to get started.")
    api_key = st.text_input("Google API Key", type="password")
    if st.button("Save API Key"):
        if api_key:
            set_api_key(api_key)
            st.session_state.api_key_entered = True
            st.session_state.GOOGLE_API_KEY = api_key
            localS.setItem("google_api_key", api_key)  # Store API key in local storage
            st.rerun()
        else:
            st.error("Please enter an API key")
    st.stop()  # Don't show the rest of the app until API key is entered

@st.cache_resource
def initialize_cached_bot():
    return initialize_bot()

# Initialize the chatbot only after API key is set
chat_function = initialize_cached_bot()

def get_chat_function():
    return initialize_cached_bot()

def display_thinking_process(placeholder, response_data, streaming=False):
    # Handle the case where response_data is a DeltaGenerator
    if hasattr(response_data, '_is_delta_generator'):
        return
        
    container = placeholder.container()
    
    if streaming:
        # Create empty placeholders for each section
        context_placeholder = container.empty()
        tools_placeholder = container.empty()
        final_placeholder = container.empty()
        return {
            'context': context_placeholder,
            'tools': tools_placeholder,
            'final': final_placeholder
        }
    else:
        # Original non-streaming display logic
        with container:
            if isinstance(response_data, dict):
                if "chat_history" in response_data:
                    with st.expander("💬 Conversation Context", expanded=False):
                        for message in response_data["chat_history"]:
                            is_human = message.type == 'human'
                            st.markdown(f"{'👤' if is_human else '🤖'} **{'User' if is_human else 'Assistant'}:** {message.content}")
                
                if "context_response" in response_data:
                    with st.expander("📚 Knowledge Base Results", expanded=True):
                        st.markdown(response_data["context_response"])
                        if "source_docs" in response_data:
                            st.divider()
                            st.markdown("**Sources:**")
                            for doc in response_data["source_docs"]:
                                st.markdown(f"- {doc.metadata.get('source', 'Unknown source')}")
                
                if "tool_response" in response_data:
                    with st.expander("🛠️ Thinking Process", expanded=True):
                        tool_data = response_data["tool_response"]
                        if "thoughts" in tool_data:
                            for i, step in enumerate(tool_data["thoughts"], 1):
                                st.markdown(f"**Step {i}:**")
                                action = step[0]
                                result = step[1]
                                st.markdown(f"🤔 **Action:** {action.tool}")
                                st.markdown(f"📥 **Input:** {action.tool_input}")
                                st.markdown(f"📤 **Result:** {result}")
                                st.divider()

def stream_update(placeholders, section, content):
    if section == 'context':
        placeholders['context'].markdown("📚 **Knowledge Base Processing:**\n" + content)
    elif section == 'tools':
        placeholders['tools'].markdown("🛠️ **Tool Processing:**\n" + content)
    elif section == 'final':
        placeholders['final'].markdown("🤖 **Final Response:**\n" + content)

def get_response(prompt, placeholder):
    chat = get_chat_function()
    try:
        # Create streaming placeholders
        placeholders = display_thinking_process(placeholder, {}, streaming=True)
        
        # Start processing with updates
        stream_update(placeholders, 'context', "Searching knowledge base...")
        result = chat(prompt)
        
        # Update context results
        if 'full_log' in result and 'context_response' in result['full_log']:
            stream_update(placeholders, 'context', result['full_log']['context_response'])
        
        # Update tool processing
        if 'full_log' in result and 'tool_response' in result['full_log']:
            tool_text = "Processing with tools...\n"
            for step in result['full_log']['tool_response'].get('thoughts', []):
                tool_text += f"- Using {step[0].tool}...\n"
                stream_update(placeholders, 'tools', tool_text)
        
        # Show final response
        stream_update(placeholders, 'final', result["final_response"])
        return result["final_response"]
    except Exception as e:
        logging.error(f"Frontend error: {str(e)}")
        return "I encountered an error. Please try again."

# Check for API key before proceeding
if st.session_state.api_key_entered:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_function = initialize_cached_bot()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("thinking_process"):
                display_thinking_process(st, message["thinking_process"])

    # Handle user input
    if prompt := st.chat_input("Ask me anything about IIIT Kottayam"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            response = get_response(prompt, thinking_placeholder)
        
        # Save the complete response including thinking process
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "thinking_process": thinking_placeholder
        })
