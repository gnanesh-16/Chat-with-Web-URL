import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import SentenceSplitter  # Using SentenceSplitter
from langchain_community.vectorstores import Chroma
from langchain_gemini import GeminiEmbeddings, ChatGemini  # Updated import
from dotenv import load_load
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore_from_url(url):
  """
  Extracts text from website, splits into sentences, and creates vector store.
  """
  loader = WebBaseLoader(url)
  document = loader.load()

  # Use SentenceSplitter for a more meaningful split
  text_splitter = SentenceSplitter()
  document_chunks = text_splitter.split_documents(document)

  # Create vector store from chunks
  vector_store = Chroma.from_documents(document_chunks, GeminiEmbeddings())

  return vector_store

def get_context_retriever_chain(vector_store):
  llm = ChatGemini()  # Updated to ChatGemini

  retriever = vector_store.as_retriever()

  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])

  retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

  return retriever_chain

def get_conversational_rag_chain(retriever_chain):
  llm = ChatGemini()  # Updated to ChatGemini

  prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
  ])

  stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

  return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
  retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
  conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

  response = conversation_rag_chain.invoke({
      "chat_history": st.session_state.chat_history,
      "input": user_input
  })

  return response['answer']

# App configuration
st.set_page_config(page_title="Chat with websites", page_icon="")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
  st.header("Settings")
  website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
  st.info("Please enter a website URL")

else:
  # Session state
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
  if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(website_url)

  # User input
  user_query = st.chat_input("Type your message here...")
  if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
  for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):  # Added closing parenthesis
        st.write(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("User"):  # Added closing parenthesis and "User" argument
        st.write(message.content)



##############worked small error ok respone_done

# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# import os
# import requests

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Set up headers for the Gemini API
# headers = {
#     'Authorization': f'Bearer {GEMINI_API_KEY}',
#     'Content-Type': 'application/json'
# }

# # def get_vectorstore_from_url(url):
# #     loader = WebBaseLoader(url)
# #     document = loader.load()

# #     text_splitter = RecursiveCharacterTextSplitter()
# #     document_chunks = text_splitter.split_documents(document)

#     # Get embeddings for each chunk
#     # document_embeddings = []
#     # for chunk in document_chunks:
#     #     response = requests.post(
#     #         'https://api.gemini.com/v1/embeddings',  # Hypothetical endpoint
#     #         headers=headers,
#     #         # json={'text': chunk['content']}
#     #         # json={'text': chunk.content}
#     #           json={'text': chunk}
#     #     )
#     #     response_data = response.json()
#     #     document_embeddings.append(response_data['embedding'])
# def get_vectorstore_from_url(url):
#     loader = WebBaseLoader(url)
#     document = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)

#     # Get embeddings for each chunk
#     document_embeddings = []
#     for chunk in document_chunks:
#         # Process chunk to extract text content appropriately
#         text_content = chunk.text  # Adjust based on actual structure of chunk
#         response = requests.post(
#             'https://api.gemini.com/v1/embeddings',  # Hypothetical endpoint
#             headers=headers,
#             json={'text': text_content}  # Pass text content for embedding generation
#         )
#         response_data = response.json()
#         document_embeddings.append(response_data['embedding'])

#     # Create vector store from embeddings
#     vector_store = Chroma.from_documents(document_chunks, document_embeddings)
#     return vector_store
    

# def get_context_retriever_chain(vector_store):
#     def retriever(prompt):
#         response = requests.post(
#             'https://api.gemini.com/v1/chat',  # Hypothetical endpoint
#             headers=headers,
#             json={'prompt': prompt}
#         )
#         return response.json()['response']
    
#     return retriever

# def get_conversational_rag_chain(retriever):
#     def conversation_chain(input, chat_history):
#         prompt = f"Given the following conversation history:\n{chat_history}\nAnswer the question: {input}"
#         return retriever(prompt)
    
#     return conversation_chain

# def get_response(user_input):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
#     response = conversation_rag_chain(user_input, st.session_state.chat_history)
    
#     return response

# # App configuration
# st.set_page_config(page_title="Chat with websites", page_icon="🤖")
# st.title("Chat with websites")

# # Sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if not website_url:
#     st.info("Please enter a website URL")
# else:
#     # Initialize session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hello, I am a bot. How can I help you?"),
#         ]
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = get_vectorstore_from_url(website_url)

#     # User input
#     user_query = st.chat_input("Type your message here...")
#     if user_query:
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))

#     # Display conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)




#%%#%%%%%%%%%%%%% bit worked
# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from gemini_api_client import GeminiEmbeddings, ChatGemini  # Hypothetical Gemini API client
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# import os

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# def get_vectorstore_from_url(url):
#     loader = WebBaseLoader(url)
#     document = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)
    
#     vector_store = Chroma.from_documents(document_chunks, GeminiEmbeddings(api_key=GEMINI_API_KEY))

#     return vector_store

# def get_context_retriever_chain(vector_store):
#     llm = ChatGemini(api_key=GEMINI_API_KEY)
    
#     retriever = vector_store.as_retriever()
    
#     prompt = ChatPromptTemplate.from_messages([
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#       ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
#     ])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
#     return retriever_chain

# def get_conversational_rag_chain(retriever_chain): 
    
#     llm = ChatGemini(api_key=GEMINI_API_KEY)
    
#     prompt = ChatPromptTemplate.from_messages([
#       ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#     ])
    
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_input):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": st.session_state.chat_history,
#         "input": user_input
#     })
    
#     return response['answer']

# # app config
# st.set_page_config(page_title="Chat with websites", page_icon="🤖")
# st.title("Chat with websites")

# # sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if website_url is None or website_url == "":
#     st.info("Please enter a website URL")

# else:
#     # session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hello, I am a bot. How can I help you?"),
#         ]
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = get_vectorstore_from_url(website_url)    

#     # user input
#     user_query = st.chat_input("Type your message here...")
#     if user_query is not None and user_query != "":
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))

#     # conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)











# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_gemini import GeminiEmbeddings, ChatGemini  # Updated import
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# import os

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# def get_vectorstore_from_url(url):
#     loader = WebBaseLoader(url)
#     document = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)
    
#     vector_store = Chroma.from_documents(document_chunks, GeminiEmbeddings(api_key=GEMINI_API_KEY))  # Updated to GeminiEmbeddings with API key

#     return vector_store

# def get_context_retriever_chain(vector_store):
#     llm = ChatGemini(api_key=GEMINI_API_KEY)  # Updated to ChatGemini with API key
    
#     retriever = vector_store.as_retriever()
    
#     prompt = ChatPromptTemplate.from_messages([
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#       ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
#     ])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
#     return retriever_chain

# def get_conversational_rag_chain(retriever_chain): 
    
#     llm = ChatGemini(api_key=GEMINI_API_KEY)  # Updated to ChatGemini with API key
    
#     prompt = ChatPromptTemplate.from_messages([
#       ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#     ])
    
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_input):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": st.session_state.chat_history,
#         "input": user_input
#     })
    
#     return response['answer']

# # app config
# st.set_page_config(page_title="Chat with websites", page_icon="🤖")
# st.title("Chat with websites")

# # sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if website_url is None or website_url == "":
#     st.info("Please enter a website URL")

# else:
#     # session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hello, I am a bot. How can I help you?"),
#         ]
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = get_vectorstore_from_url(website_url)    

#     # user input
#     user_query = st.chat_input("Type your message here...")
#     if user_query is not None and user_query != "":
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))

#     # conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)







# # pip install streamlit langchain langchain-gemini beautifulsoup4 python-dotenv chromadb


# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_gemini import GeminiEmbeddings, ChatGemini  # Updated import
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# load_dotenv()

# def get_vectorstore_from_url(url):
#     # get the text in document form
#     loader = WebBaseLoader(url)
#     document = loader.load()
    
#     # split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)
    
#     # create a vectorstore from the chunks
#     vector_store = Chroma.from_documents(document_chunks, GeminiEmbeddings())  # Updated to GeminiEmbeddings

#     return vector_store

# def get_context_retriever_chain(vector_store):
#     llm = ChatGemini()  # Updated to ChatGemini
    
#     retriever = vector_store.as_retriever()
    
#     prompt = ChatPromptTemplate.from_messages([
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#       ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
#     ])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
#     return retriever_chain

# def get_conversational_rag_chain(retriever_chain): 
    
#     llm = ChatGemini()  # Updated to ChatGemini
    
#     prompt = ChatPromptTemplate.from_messages([
#       ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#       MessagesPlaceholder(variable_name="chat_history"),
#       ("user", "{input}"),
#     ])
    
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_input):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": st.session_state.chat_history,
#         "input": user_input
#     })
    
#     return response['answer']

# # app config
# st.set_page_config(page_title="Chat with websites", page_icon="🤖")
# st.title("Chat with websites")

# # sidebar
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if website_url is None or website_url == "":
#     st.info("Please enter a website URL")

# else:
#     # session state
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#             AIMessage(content="Hello, I am a bot. How can I help you?"),
#         ]
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = get_vectorstore_from_url(website_url)    

#     # user input
#     user_query = st.chat_input("Type your message here...")
#     if user_query is not None and user_query != "":
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))

#     # conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)
