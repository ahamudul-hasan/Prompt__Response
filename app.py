import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit Cloud Secrets (for cloud deployment)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="Gemini Chat",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
<style>
    /* Clean header styling */
    .chat-header {
        text-align: center;
        padding-bottom: 1.5rem;
    }
    .chat-header h1 {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        color: #888888;
        font-size: 0.95rem;
    }
    /* Suggestion pills styling */
    .stButton > button {
        border-radius: 20px;
        font-size: 0.85rem;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("✨ Gemini Chat")
    st.markdown("A Gemini conversational app with multi-turn memory.")
    
    st.divider()
    
    # Model configuration
    temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Title
st.markdown("""
<div class="chat-header">
    <h1>✨ Gemini Assistant</h1>
    <div class="chat-subtitle">Ask anything, follow up with questions, and enjoy natural multi-turn conversations.</div>
</div>
""", unsafe_allow_html=True)

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome suggestions if conversation is empty
if len(st.session_state.messages) == 0:
    st.info("👋 **Welcome!** Type a question below or choose an example topic to start:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Explain quantum computing simply", use_container_width=True):
            st.session_state.preset_prompt = "Explain quantum computing simply"
            st.rerun()
    with col2:
        if st.button("🚀 Why is the sky blue?", use_container_width=True):
            st.session_state.preset_prompt = "Why is the sky blue?"
            st.rerun()

# Display existing messages from the conversation history
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Check if a preset button was clicked
user_query = None
if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    user_query = st.session_state.preset_prompt
    del st.session_state["preset_prompt"]
else:
    user_query = st.chat_input("Ask a question or continue the conversation...")

# Handle new user input
if user_query:
    # 1. Display user query immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # 2. Build history of previous turns for LangChain
    history_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    # 3. Append current user message to session state
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 4. Construct LangChain Prompt Template with History
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful, knowledgeable AI assistant. "
            "Give clear, concise, and structured answers. "
            "Maintain conversational context from previous messages."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # 5. Initialize Gemini model (high-speed flash-lite)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=temperature,
        max_tokens=800
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    # 6. Stream the assistant response
    with st.chat_message("assistant", avatar="✨"):
        try:
            response = st.write_stream(
                chain.stream({
                    "history": history_messages,
                    "question": user_query
                })
            )
            # 7. Append assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error communicating with Gemini: {e}")