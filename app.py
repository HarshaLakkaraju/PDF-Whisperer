# Import necessary libraries
from pyngrok import ngrok  # For creating secure tunnels to localhost
import subprocess  # To run external processes
import streamlit as st  # For building web applications
from PyPDF2 import PdfReader  # For reading and extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into manageable chunks
from langchain_core.prompts import ChatPromptTemplate  # To create templates for chat prompts
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings  # For text embeddings using SpaCy
from langchain_community.vectorstores import FAISS  # For creating vector stores with FAISS
from langchain.tools.retriever import create_retriever_tool  # For creating retrieval tools
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain_anthropic import ChatAnthropic  # For using Anthropic's language models (commented out)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # For using OpenAI's language models and embeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent  # To create agents that can use tools
from litellm import completion
import os  # For operating system functionalities

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize SpaCy embeddings model
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    """
    Reads and extracts text from a list of PDF documents.
    
    Args:
        pdf_doc (list): List of uploaded PDF documents.
    
    Returns:
        str: Extracted text from all pages in the PDF documents.
    """
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    """
    Splits text into smaller chunks for processing.
    
    Args:
        text (str): The full text extracted from PDF documents.
    
    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    """
    Creates a vector store from text chunks using FAISS and saves it locally.
    
    Args:
        text_chunks (list): List of text chunks to be vectorized and stored.
    """

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    """
    Creates a conversational agent chain to answer questions using the given tools.
    
    Args:
        tools: The tools used for retrieving and processing information.
        ques (str): The question input by the user.
    """
    # The Anthropic API key setup is commented out
    # os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"), verbose=True)
    
    # Using OpenAI's GPT model instead
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key="")

    #run this model in the local system
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key="anything",base_url="http://0.0.0.0:4000")

    # to run this model in local system  using ollama + litellm 

    # download litellm >> pip install 'litellm[proxy]'
    # download ollama >> https://ollama.com/download/windows

    # pull any open source llm model using ollama >> https://ollama.com/library >> ollama pull <model_name>
    
    # start litellm in cmd >>  litellm --model ollama/<model_name> &
    #import litellm into the code 
    # copy url in place of "base_url"


    # Define the prompt template for the conversational agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tool = [tools]  # List of tools for the agent
    agent = create_tool_calling_agent(llm, tool, prompt)  # Create the agent with the LLM and tools
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)  # Create an executor for the agent
    response = agent_executor.invoke({"input": ques})  # Invoke the agent to get a response
    print(response)  # Print the response to the console
    st.write("Reply: ", response['output'])  # Display the response in Streamlit

def user_input(user_question):
    """
    Handles user input for the PDF-based Q&A system.
    
    Args:
        user_question (str): The question input by the user.
    """
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)  # Load the local FAISS database
    retriever = new_db.as_retriever()  # Convert the FAISS database to a retriever
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")  # Create a retrieval tool
    get_conversational_chain(retrieval_chain, user_question)  # Use the retrieval tool in the conversational chain

def main():
    """
    Main function to run the Streamlit app for RAG-based chat with PDF.
    """
    st.set_page_config("Chat PDF")  # Set the Streamlit page configuration
    st.header("RAG based Chat with PDF")  # Display the header of the app

    user_question = st.text_input("Ask a Question from the PDF Files")  # Input box for user questions

    if user_question:
        user_input(user_question)  # Process the user input if a question is provided

    with st.sidebar:  # Sidebar for file upload and processing
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)  # File uploader for PDFs
        if st.button("Submit & Process"):  # Button to submit and process the PDFs
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)  # Read and extract text from PDFs
                text_chunks = get_chunks(raw_text)  # Split text into chunks
                vector_store(text_chunks)  # Create and save the vector store
                st.success("Done")  # Display success message

if __name__ == "__main__":
    main()  # Run the main function if the script is executed
