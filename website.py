import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Load the Hugging Face conversational model
conversational_pipeline = pipeline("conversational", model="facebook/blenderbot-400M-distill")

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        texts = soup.stripped_strings
        return " ".join(texts)
    except Exception as e:
        return str(e)

# Initialize LangChain memory
memory = ConversationBufferMemory()

# Initialize LangChain conversational chain
chain = ConversationChain(memory=memory, llm=conversational_pipeline)

# Streamlit UI
st.title("Chat with Any Website")

# Input URL
url = st.text_input("Enter the URL of the website you want to chat with:")

if url:
    # Scrape website content
    website_content = scrape_website(url)
    st.write("Website content scraped successfully.")
    
    # Display scraped content (optional)
    if st.checkbox("Show scraped content"):
        st.write(website_content)

    # Initialize conversation
    conversation_history = []

    # User input
    user_input = st.text_input("You:")

    if st.button("Send"):
        # Append user input to conversation history
        conversation_history.append(f"User: {user_input}")

        # Generate response using LangChain
        chain_output = chain.run(input_text=user_input, context=website_content)

        # Append model response to conversation history
        conversation_history.append(f"Bot: {chain_output}")

        # Display conversation history
        st.write("\n".join(conversation_history))
