from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = "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" ## find at platform.openai.com
pinecone_api_key = "d230654b-0530-4109-a7f8-d6d83d952e62"
model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'chatbot' 

if index_name not in pc.list_indexes().names():
    # Assuming the vector dimension of 'all-MiniLM-L6-v2' model embeddings
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp',  # or 'aws', based on where you created your Pinecone project
            region='us-central1'  # Make sure this matches your project's region
        )
    )

index = pc.Index(name=index_name)

# pinecone.init(api_key='d230654b-0530-4109-a7f8-d6d83d952e62', # find at app.pinecone.io
#               environment='gcp-starter' # next to api key in console
#              )
# index = pinecone.Index('chatbot')

# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query([input_em], top_k=2)
#     # Construct a response based on the metadata of the matched results
#     response_texts = [match['metadata']['text'] for match in result['matches'][0]]
#     return "\n".join(response_texts)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2)
    matches = result['matches']
    
    # Initialize an empty list to store texts from matches
    texts = []
    
    # Iterate over matches and collect their text
    for match in matches:
        if 'metadata' in match and 'text' in match['metadata']:
            texts.append(match['metadata']['text'])
    
    # Join the texts with a newline, or return a default message if there were no matches
    if texts:
        return "\n".join(texts)
    else:
        return "No matches found."
    
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Make sure this is a chat-compatible model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{conversation}\n{query}"}
        ]
    )
    # Adjust the response extraction based on the structure of chat completion responses
    return response['choices'][0]['message']['content']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
