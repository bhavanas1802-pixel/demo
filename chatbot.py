from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

print("Chatbot ready! Type 'exit' to stop.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break

#     response = client.chat.completions.create(
#         model=deployment_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": user_input}
#         ]
#     )

     print("Bot:", response.choices[0].message.content)

#st.set_page_config(page_title="Azure OpenAI Chatbot")

st.title("OpenAI Chatbot")

# Validate env vars
required_env = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]

missing = [k for k in required_env if not os.getenv(k)]
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Azure OpenAI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=deployment_name,
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=300,
            )

            reply = response.choices[0].message.content
            st.markdown(reply)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
