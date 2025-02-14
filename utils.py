from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = "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" ## find at platform.openai.com
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='d230654b-0530-4109-a7f8-d6d83d952e62',
              environment='gcp-starter' 
             )

index = pinecone.Index('chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-embedding-ada-002",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']
#     response = client.chat.completions.create(
#         model = "gpt-3.5-turbo",
#         # prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG:
#         # \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#         messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ],
#         temperature=0.3,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     return response['choices'][0]

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
