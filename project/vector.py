import json
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()
# Supabase credentials
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)

# Load JSON data from the data folder
with open('data/players.json', 'r') as f:
    players = json.load(f)

# Insert the data into Supabase table
response = supabase.table("hockey_players").insert(players).execute()

print("Upload Status:", response)


client = OpenAI()

players = supabase.table('hockey_players').select('id, biography').execute().data

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

for player in players:
    response = client.embeddings.create(
    input=player['biography'],
    model="text-embedding-ada-002")
    langchain_response=embeddings.embed_query(player['biography'])
    
    
    vector=response.data[0].embedding
    supabase.table('hockey_players').update({'bio_vector': vector}).eq('id', player['id']).execute()
    supabase.table('hockey_players').update({'langchain_bio_vector': langchain_response}).eq('id', player['id']).execute()

# Embed the user query
question = "Which player was the captain multiple times?"
query_embedding = client.embeddings.create(
    input=question,
    model="text-embedding-ada-002"
).data[0].embedding

# Add debug prints
print("Query embedding length:", len(query_embedding))

# Call the stored function with adjusted threshold
response = supabase.rpc('find_embedding', {
    'table_name': 'hockey_players',
    'query_embedding_field': 'bio_vector',
    'query_embedding': query_embedding,
    'match_threshold': 0.5,  # Lowered threshold for testing
    'match_count': 5  # Increased count for testing
}).execute()

# Debug print
print("OpenAI Search Response:", response.data)

# Only proceed if we have results
if not response.data:
    print("No matches found in OpenAI search")
else:
    for result in response.data:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}")

    matched_ids = list(set([player['id'] for player in response.data]))
    print("Matched IDs:", matched_ids)
    
    players_bio = supabase.table('hockey_players').select('id,name,biography').in_('id', matched_ids).execute().data
    print("Players found:", len(players_bio))

# Modified context building to include player name and ensure unique biographies
context = ""
for player in players_bio:
    context += f"Player {player['name']}'s Biography: {player['biography']}\n\n"
    
print(context)
 
# Now, context holds the concatenated biographies of the top 3 players

# Example: Now pass this context to the LLM
question = "Which player was the captain multiple times?"
rag_prompt = f"""
Using the following player biographies, answer the question: {question}
{context}
Provide your answer based only on this information.
"""
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": rag_prompt
        }
    ]
)

print("response using openai : ",completion.choices[0].message.content)

# LangChain search
query_embedding_langchain = embeddings.embed_query(question)

# Debug print
print("LangChain embedding length:", len(query_embedding_langchain))

response_langchain = supabase.rpc('find_embedding', {
    'table_name': 'hockey_players',
    'query_embedding_field': 'langchain_bio_vector',
    'query_embedding': query_embedding_langchain,
    'match_threshold': 0.5,  # Lowered threshold for testing
    'match_count': 5  # Increased count for testing
}).execute()

# Debug print
print("LangChain Search Response:", response_langchain.data)

# Only proceed if we have results
if not response_langchain.data:
    print("No matches found in LangChain search")
else:
    for result in response_langchain.data:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.4f}")

    matched_ids_langchain = list(set([player['id'] for player in response_langchain.data]))
    print("LangChain Matched IDs:", matched_ids_langchain)
    
    players_bio_langchain = supabase.table('hockey_players').select('id,name,biography').in_('id', matched_ids_langchain).execute().data
    print("LangChain Players found:", len(players_bio_langchain))

# Similarly update the langchain context building
context_langchain = ""
for player in players_bio_langchain:
    context_langchain += f"Player {player['name']}'s Biography: {player['biography']}\n\n"

# Now, context holds the concatenated biographies of the top 3 players

# Example: Now pass this context to the LLM
question = "Which player was the captain multiple times?"
rag_prompt = f"""
Using the following player biographies, answer the question: {question}
{context_langchain}
Provide your answer based only on this information.
"""

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

messages = [
    ("human", rag_prompt),
]
ai_msg = llm.invoke(messages)
print("response using langchain : ",ai_msg.content)

# Add this after the vector storage loop
test_record = supabase.table('hockey_players').select('bio_vector,langchain_bio_vector').limit(1).execute()
if test_record.data:
    print("OpenAI vector length:", len(test_record.data[0]['bio_vector']))
    print("LangChain vector length:", len(test_record.data[0]['langchain_bio_vector']))