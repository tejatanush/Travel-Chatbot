from flask import Flask, render_template, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Setting up API key
api_key = os.getenv("HuggingFaceHub_API_TOKEN")
if not api_key:
    raise ValueError("Missing Hugging Face API token. Please set it in your .env file.")

os.environ["HuggingFaceHub_API_TOKEN"] = api_key

# Initialize Hugging Face LLM
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id=model_id, max_length=500, temperature=0.5)  # Reduced max_length for better response formatting

# Global variables
itinerary = ""
chat_enabled = False

# Route for main page
@app.route("/")
def home():
    global itinerary, chat_enabled
    itinerary = ""  
    chat_enabled = False
    return render_template("index.html", itinerary=itinerary, chat_enabled=chat_enabled)

# Route to generate itinerary
@app.route("/generate_itinerary", methods=["POST"])
def generate_itinerary():
    global itinerary, chat_enabled
    
    data = request.json  # Use JSON for better handling
    source = data.get("source", "").strip()
    destination = data.get("destination", "").strip()
    num_days = data.get("num_days", "").strip()

    if not source or not destination or not num_days:
        return jsonify({"error": "Missing required fields"}), 400

    # Define the prompt
    prompt = PromptTemplate(
        input_variables=["source", "destination", "num_days"],
        template=(
            "Generate a {num_days}-day travel itinerary for a trip from {source} to {destination}."
            " Provide a detailed plan for each day, including:"
            "\n- Morning activities"
            "\n- Afternoon activities"
            "\n- Evening activities"
            "\n- Recommendations for restaurants and cafes"
            "\n- Any travel tips for the day.\n\n"
            "Please follow this format:\n\n"
            "Day 1:\n- Morning: [Description]\n- Afternoon: [Description]\n- Evening: [Description]\n"
            "Recommended Restaurants: [List]\nTravel Tips: [Tip]\n\n"
            "Continue this format for all days until Day {num_days}."
        )
    )

    # Run LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"source": source, "destination": destination, "num_days": num_days})

    itinerary = response.get("text", "").strip()  # Extract text safely
    
    if not itinerary:
        return jsonify({"error": "Failed to generate itinerary. Try again later."}), 500

    chat_enabled = True  # Enable chat
    return jsonify({"itinerary": itinerary, "chat_enabled": chat_enabled})

# Route for chatbot conversation
@app.route("/chat", methods=["POST"])
def chat():
    global chat_enabled

    if not chat_enabled:
        return jsonify({"response": "Please generate an itinerary first."})

    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    response = llm.invoke(user_message)  # Use `.invoke()` for correct API usage
    return jsonify({"response": response.get("text", "I'm not sure how to respond.")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
