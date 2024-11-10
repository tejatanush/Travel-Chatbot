from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Setting up the API key and model
api_key = os.getenv("HuggingFaceHub_API_TOKEN")
os.environ["HuggingFaceHub_API_TOKEN"] = api_key
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id=model_id, max_length=10000, temperature=0.5)

# Initialize global variables to store itinerary and chat state
itinerary = ""
chat_enabled = False  # To track if chat is allowed

# Route for the main page
@app.route("/")
def home():
    global itinerary, chat_enabled
    itinerary = ""  # Reset itinerary on page load
    chat_enabled = False  # Disable chat initially
    return render_template("index.html", itinerary=itinerary, chat_enabled=chat_enabled)

# Route to handle the form submission
@app.route("/generate_itinerary", methods=["POST"])
def generate_itinerary():
    global itinerary, chat_enabled
    source = request.form.get("source")
    destination = request.form.get("destination")
    num_days = request.form.get("num_days")

    # Define the prompt for itinerary generation
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
    
    
    # Run the chain to generate itinerary
    chain = LLMChain(llm=llm, prompt=prompt)
    itinerary = chain.run({"source": source, "destination": destination, "num_days": num_days})
    
    # Enable chat and send the itinerary back to the frontend
    chat_enabled = True
    return jsonify({"itinerary": itinerary, "chat_enabled": chat_enabled})

# Route to handle follow-up chat messages
@app.route("/chat", methods=["POST"])
def chat():
    if not chat_enabled:
        return jsonify({"response": "Please provide trip details first."})
    
    user_message = request.json.get("message")
    response = llm(user_message)  # Get response from the LLM for follow-up questions
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
