from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from datetime import datetime
import logging
import json
from pathlib import Path

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYew4okjx4jmR7xbKhLj2mAckgtUUbR-k"

llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-2.0-flash-exp",
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize ChromaDB
PERSIST_DIRECTORY = "db"
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

DB_TRACKING_FILE = "db_files.json"

def load_db_tracking():
    """Load the database tracking information"""
    if os.path.exists(DB_TRACKING_FILE):
        with open(DB_TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_db_tracking(tracking_data):
    """Save the database tracking information"""
    with open(DB_TRACKING_FILE, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def add_file(file_path):
    """Add a single file to the vector database and track its IDs"""
    try:
        # Load existing tracking data
        tracking_data = load_db_tracking()
        
        # Load and split the document
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Add to vectorstore and get IDs
        ids = vectorstore.add_documents(texts)
        vectorstore.persist()
        
        # Track the file and its chunk IDs
        tracking_data[str(Path(file_path).resolve())] = {
            'chunk_ids': ids,
            'added_date': datetime.now().isoformat()
        }
        save_db_tracking(tracking_data)
        
        return f"Added {len(texts)} chunks from {file_path}"
    except Exception as e:
        return f"Error adding file: {str(e)}"

def remove_file(file_path):
    """Remove a file's chunks from the vector database"""
    try:
        tracking_data = load_db_tracking()
        resolved_path = str(Path(file_path).resolve())
        
        if resolved_path not in tracking_data:
            return f"File {file_path} not found in tracking data"
        
        # Delete chunks from vectorstore
        chunk_ids = tracking_data[resolved_path]['chunk_ids']
        vectorstore.delete(chunk_ids)
        vectorstore.persist()
        
        # Remove from tracking
        del tracking_data[resolved_path]
        save_db_tracking(tracking_data)
        
        return f"Removed file {file_path} and its {len(chunk_ids)} chunks"
    except Exception as e:
        return f"Error removing file: {str(e)}"

def list_tracked_files():
    """List all tracked files and their chunks in the database"""
    tracking_data = load_db_tracking()
    return {
        path: {
            'chunk_count': len(data['chunk_ids']),
            'added_date': data['added_date']
        }
        for path, data in tracking_data.items()
    }

def add_directory(dir_path):
    """Add all text files from a directory to the vector database"""
    loader = DirectoryLoader(dir_path, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vectorstore.add_documents(texts)
    vectorstore.persist()
    return f"Added {len(texts)} chunks from directory {dir_path}"


def search_knowledge_base(query, k=3):
    """Search the vector database for relevant context"""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def clear_database():
    """Clear all documents from the vector database"""
    vectorstore.delete_collection()
    vectorstore.persist()
    return "Database cleared"

def list_documents():
    """List all documents in the vector database with their metadata"""
    try:
        collection = vectorstore.get()
        if not collection['ids']:
            return "Database is empty"
        
        results = []
        for i, doc_id in enumerate(collection['ids']):
            doc = collection['documents'][i]
            metadata = collection['metadatas'][i]
            results.append({
                'id': doc_id,
                'content_preview': doc[:200] + '...' if len(doc) > 200 else doc,
                'metadata': metadata
            })
        return results
    except Exception as e:
        return f"Error listing documents: {str(e)}"

def delete_document(doc_id):
    """Delete a specific document from the vector database"""
    try:
        vectorstore.delete([doc_id])
        vectorstore.persist()
        return f"Deleted document {doc_id}"
    except Exception as e:
        return f"Error deleting document: {str(e)}"

def get_document_by_id(doc_id):
    """Retrieve a specific document's full content by ID"""
    try:
        collection = vectorstore.get([doc_id])
        if collection['ids']:
            return {
                'id': doc_id,
                'content': collection['documents'][0],
                'metadata': collection['metadatas'][0]
            }
        return "Document not found"
    except Exception as e:
        return f"Error retrieving document: {str(e)}"

# Define tools
with open("./data/premade/mess_menu.txt", "r") as file:
    mess_menu = file.read()
with open("./data/premade/inst_ calender.txt", "r") as file:
    inst_calendar = file.read()

def load_mess_menu(input_str=None):
    """Load the mess menu context into the conversation"""
    return f"This is the mess menu of the campus. Please use this context to answer queries about the menu: {mess_menu}"

def load_calendar(input_str=None):
    """Load the institute calendar context into the conversation"""
    return f"This is the academic calendar of IIIT Kottayam. Please use this context to answer queries about academic dates and holidays: {inst_calendar}"

def get_time_context(arg):
    """Get current date and time information"""
    print(arg)
    now = datetime.now()
    return f"""Current time context:
- Date: {now.strftime('%A, %B %d, %Y')}
- Time: {now.strftime('%I:%M %p')}
- Day of week: {now.strftime('%A')}"""

tools = [
    Tool(
        name="Load Mess Menu",
        func=load_mess_menu,
        description="Use this tool to load the mess menu context when the query is related to food, mess, or menu.",
    ),
    Tool(
        name="Load Academic Calendar",
        func=load_calendar,
        description="Use this tool to load the academic calendar when the query is related to academic dates, holidays, or institute events like exams.",
    ),
    Tool(
        name="Get Date and Time Context",
        func=get_time_context,
        description="Use this tool to get the current date and time information and what day it is"
    )
]

# Set up logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    allow_multiple_tools=True  # Enable multiple tools
)

# Define the conversation template
template = """You are an AI assistant for IIIT Kottayam students.
Current conversation:
{chat_history}
Human: {question}
Assistant:"""

prompt = PromptTemplate(input_variables=["chat_history", "question"], template=template)

# Initialize the retrieval chain components
retrieval_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
    memory=retrieval_memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True
)

# Modify chat function to handle errors
def chat(user_input):
    try:
        logging.info(f"User Input: {user_input}")
        
        # Get context-based response
        retrieval_response = retrieval_chain({"question": user_input})
        context_response = retrieval_response["answer"]
        logging.info(f"Context Response: {context_response}")
        
        # Get tool response
        tool_response = agent.run(user_input)
        logging.info(f"Tool Response: {tool_response}")
        
        # Clean up and combine responses
        responses = [r.strip() for r in [context_response, tool_response] if r and r.strip()]
        final_response = "\n\n".join(responses)
        logging.info(f"Final Response: {final_response}")
        
        return {
            "final_response": final_response.split("Final Answer:")[-1].strip() if "Final Answer:" in final_response else final_response,
            "full_log": {
                "context_response": context_response,
                "tool_response": tool_response,
                "final_response": final_response
            }
        }
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        if "429" in str(e):
            return {"final_response": "Sorry, I'm a bit busy right now. Please try again in a moment."}
        return {"final_response": f"I encountered an error: {str(e)}"}

def initialize_bot():
    """Initialize the chatbot and return the chat function"""
    if not os.path.exists(PERSIST_DIRECTORY):
        # Add individual files instead of entire directory
        data_files = [
            "./data/mess_menu.txt",
            "./data/inst_calender.txt",
            "./data/faculty_details.txt",
            "./data/caricululm.txt"
        ]
        for file_path in data_files:
            if os.path.exists(file_path):
                result = add_file(file_path)
                logging.info(f"Added file {file_path}: {result}")
    return chat

if __name__ == "__main__":
    initialize_bot()
    
    # List tracked files
    print("\nTracked files in database:")
    tracked_files = list_tracked_files()
    for file_path, info in tracked_files.items():
        print(f"\nFile: {file_path}")
        print(f"Chunks: {info['chunk_count']}")
        print(f"Added: {info['added_date']}")
    
    # Normal chat loop
    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat(user_input)
        print("Assistant:", response["final_response"])