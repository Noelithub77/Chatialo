import os
import json
import logging
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from vdb_management import vdb


def set_api_key(key):
    """Set the Google API key"""
    os.environ["GOOGLE_API_KEY"] = key

os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"


# Enhanced logging configuration
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add a function to handle structured logging
def log_interaction(interaction_type, data):
    """Log interaction details in a structured format"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": interaction_type,
        "data": data
    }
    logging.info(json.dumps(log_entry, default=str))

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-1.5-flash-8b",
    # model="gemini-2.0-flash-exp",
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)


vdb_obj = vdb(persist_directory="db")


# Don't explain this
def load_context_file(filename):
    try:
        with open(f"./data/{filename}", "r") as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return ""

# Initialize memory for retrieval chain
chatHistory = ConversationBufferMemory()

# Define conversation template
template = """You are Mr.G, an AI assistant for IIIT Kottayam students.Give me detailed explanation
Context: {context}
Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=template
)


retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vdb_obj.get_retriever(),
    memory=chatHistory,
    get_chat_history=lambda h: str(h),
    combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
    return_source_documents=True,
    verbose=True
)



def get_retrieval_response(query):
    """Tool function to get response from retrieval chain"""
    response = retrieval_chain({"question": query})
    return f"Knowledge Base Response: {response['answer']}\nSources: {[doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]}"

# Updated tools list with retrieval tool
# Don't Explain tools in detail
tools = [
    Tool(
        name="Knowledge Base Search",
        func=get_retrieval_response,
        description="Use this tool to search the knowledge base for specific information about IIIT Kottayam"
    ),
    Tool(
        name="Load Mess Menu",
        func=lambda x: f"Mess menu context: {load_context_file('mess_menu.txt')}",
        description="Load mess menu context for food-related queries",
    ),
    Tool(
        name="Load Academic Calendar",
        func=lambda x: f"Academic calendar context: {load_context_file('inst_calender.txt')}",
        description="Load academic calendar context for academic dates and holidays",
    ),
    Tool(
        name="Get Date and Time Context",
        func=lambda x: f"Current time context: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}",
        description="Get current date and time information",
    ),
    Tool(
        name="Load Curriculum",
        func=lambda x: f"Curriculum context: {load_context_file('caricululm.txt')}",
        description="Load curriculum context for course-related queries",
    ),
    Tool(
        name="Load Milma Menu",
        func=lambda x: f"Milma Cafe menu context: {load_context_file('milma_menu.txt')}",
        description="Load Milma Cafe menu context for food-related queries",
    ),
    Tool(
        name="Load Faculty Details",
        func=lambda x: f"Faculty details context: {load_context_file('faculty_details.txt')}",
        description="Load faculty details context for faculty-related queries",
    )
]

# Initialize chat history
chat_history = ChatMessageHistory(session_id="iiitk-session")

# Update the prompt template with required variables
template = """You are Mr.G, an AI assistant for IIIT Kottayam students.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Update agent initialization
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

template = """You are Mr.G, an AI assistant for IIIT Kottayam students.Give me detailed explanation
Current conversation:
{chat_history}
Context: {context}
Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"], 
    template=template
)


# Start here;
def chat(user_input):
    try:
        logging.info(f"New chat interaction started")
        log_interaction("user_input", {"input": user_input})
        
        # Use agent with memory
        agent_response = agent_with_memory.invoke(
            {"input": user_input},
            {"configurable": {"session_id": "iiitk-session"}}
        )
        
        # Log the complete interaction
        log_interaction("agent_execution", {
            "input": user_input,
            "output": agent_response["output"],
            "intermediate_steps": agent_response.get("intermediate_steps", []),
            "chat_history": [str(msg) for msg in chat_history.messages],
        })
        
        # Log tool usage
        if "intermediate_steps" in agent_response:
            for step in agent_response["intermediate_steps"]:
                log_interaction("tool_usage", {
                    "tool": step[0].tool,
                    "input": step[0].tool_input,
                    "output": step[1]
                })
        
        response = {
            "final_response": agent_response["output"],
            "full_log": {
                "thoughts": agent_response.get("intermediate_steps", []),
                "chat_history": chat_history.messages,
                "context_response": "",
                "tool_response": {"thoughts": agent_response.get("intermediate_steps", [])}
            }
        }
        
        # Log final response
        log_interaction("final_response", response)
        
        return response
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "user_input": user_input
        }
        log_interaction("error", error_details)
        logging.error(f"Chat error: {str(e)}", exc_info=True)
        return {"final_response": f"I encountered an error: {str(e)}"}

def initialize_bot():
    """Initialize the chatbot and return the chat function"""
    # Clear both memories when initializing
    return chat


if __name__ == "__main__":
    # os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"
    initialize_bot()
    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat(user_input)
        print("Assistant:", response["final_response"])
