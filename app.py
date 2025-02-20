import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model Configuration
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=(
        "You are an expert at teaching science to kids. "
        "Your task is to engage in conversations about science and answer questions. "
        "Explain scientific concepts in a way that is easily understandable. "
        "Use analogies and relatable examples, humor, and interactive questions. "
        "Suggest ways these concepts can be related to the real world with observations and experiments."
    ),
)

# Streamlit UI setup
st.set_page_config(page_title="Science Chatbot", page_icon="🔬", layout="wide")
st.title("🔬 Science Chatbot for Kids")
st.write("Hello! Ask me anything about science, and I'll explain it in a fun way! 🧪✨")

# Initialize chat history
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    text = message["parts"][0]
    if role == "user":
        st.chat_message("You", avatar="🧑").write(text)
    else:
        st.chat_message("Bot", avatar="🤖").write(text)

# User input
user_input = st.text_input("Ask a science question:", key="user_input")
if st.button("Send") and user_input:
    # Display user message
    st.chat_message("You", avatar="🧑").write(user_input)
    
    # Get bot response
    response = st.session_state.chat_session.send_message(user_input)
    model_response = response.text
    
    # Display bot response
    st.chat_message("Bot", avatar="🤖").write(model_response)
    
    # Update chat history
    st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
    st.session_state.chat_history.append({"role": "model", "parts": [model_response]})
