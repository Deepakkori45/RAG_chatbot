from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
# Initialize OpenAI and Pinecone with API keys
GOOGLE_API_KEY = "AIzaSyC5jVGT9OHx4soEsliU60ByZsieobJPRms"
pinecone_api_key = "d230654b-0530-4109-a7f8-d6d83d952e62"

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
index_name = 'chatbot'

# Ensure the Pinecone index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the sentence transformer model embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp-starter',
            region='us-central1'
        )
    )

index = pc.Index(name=index_name)

# Function to find matches in Pinecone index
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2)
    matches = result['matches']
    
    texts = [match['metadata']['text'] for match in matches if 'metadata' in match and 'text' in match['metadata']]
    return "\n".join(texts) if texts else "No matches found."

# Function to refine query using OpenAI
def query_refiner(conversation, query):
    response = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key= GOOGLE_API_KEY,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{conversation}\n{query}"}
        ]
    )
    return response
    # return response['choices'][0]['message']['content']

# Function to get conversation string for Streamlit
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
