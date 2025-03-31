from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import requests
import os
from dotenv import load_dotenv
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Updated to use as_retriever


# Hugging Face Inference API Call
def hf_generate_response(prompt):
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"  
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": None, "temperature": 0.4}}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=40)  # Timeout set to 30 seconds

        if response.status_code == 200:
            json_response = response.json()
            if isinstance(json_response, list) and len(json_response) > 0:
                return json_response[0].get("generated_text", "Error: No response generated.")
            else:
                return "Error: Unexpected response format from API."
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait and try again."
        else:
            return f"Error: API responded with status {response.status_code}."
    
    except requests.Timeout:
        return "Error: API request timed out. Try again later."
    except requests.RequestException as e:
        return f"Error: {str(e)}"


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    # Retrieve relevant documents from Pinecone
    retrieved_docs = retriever.invoke(msg)  
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # **Ensure responses continue naturally**
    if msg.lower() in ["explain more", "tell me more", "continue", "go on"]:
        full_prompt = f"{context}\nAssistant: Continue explaining in more detail."
    else:
        full_prompt = f"{context}\nUser: {msg}\nAssistant:"

    response_text = hf_generate_response(full_prompt)

    # **Extract only the assistant's response (forcing output after "Assistant:")**
    if "Assistant:" in response_text:
        response_text = response_text.split("Assistant:", 1)[-1].strip()
    else:
        response_text = response_text.strip()

    print("Response:", response_text)

    return response_text  # Send only the refined response to frontend


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default to 8000 if not set
    app.run(host="0.0.0.0", port=port)
