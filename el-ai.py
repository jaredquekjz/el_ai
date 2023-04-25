# Read the secrets from the secrets.toml file
import toml
import streamlit as st
import subprocess
import openai

token = st.secrets["token"]
openai.api_key = st.secrets["API_KEY"]
UNIQUE_ID = st.secrets["UNIQUE_ID"]

# Run the nomic login command

command = ["nomic", "login", token]
result = subprocess.run(command, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)

if result.returncode == 0:
    print("Command executed successfully")
else:
    print("Command execution failed")

# Other requirements

import numpy as np
from nomic import atlas, AtlasProject
import pandas as pd
from scipy.spatial.distance import cdist
import textwrap
import io
from io import StringIO


# Streamlit app
st.set_page_config(
    page_title="Ask El",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":books:",
)

# Function to convert a question prompt into an embedding

def get_question_embedding(question: str) -> list[float]:
    result = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )
    return result["data"][0]["embedding"]



# Function to find the most relevant context based on the question
project = AtlasProject(project_id=UNIQUE_ID,
                       organization_name='jaredquek', modality='embedding')
map = project.maps[0]

# Change the find_relevant_context function to return a list of unique contexts


def find_relevant_context(question: str, map, project, top_n: int = 35) -> list:
    question_embedding = get_question_embedding(question)
    question_embedding_np = np.array([question_embedding])

    with project.wait_for_project_lock():
        neighbors, distances = map.vector_search(
            queries=question_embedding_np, k=top_n * 2)

    unique_neighbors = []
    unique_contexts = []
    for neighbor in neighbors[0]:
        if neighbor not in unique_neighbors:
            unique_neighbors.append(neighbor)
            unique_contexts.append(project.get_data(
                ids=[neighbor])[0]['combined'])
        if len(unique_neighbors) == top_n:
            break

    return unique_contexts

# Function to ask a question and get an answer from ChatGPT


def ask_chatgpt(question: str, relevant_context: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
                "content": "You are a literary analysis AI that knows the context provided to answer the user's questions based on the context."},
            {"role": "user", "content": f"Context: {relevant_context}"},
            {"role": "user", "content": question}
        ],
        max_tokens=1500,
        n=1,
        temperature=0.5,
    )
    return response.choices[0].message["content"].strip()

# ... rest of the imports and code ...


# ... rest of the imports and code ...

# Add CSS for custom styling
st.markdown(
    """
<style>
   .title {
       font-family: Arial, sans-serif;
       font-weight: bold;
       font-size: 40px;
       color: #FFD700;
       text-align: center;
   }
   .subtitle {
       font-family: Arial, sans-serif;
       font-size: 20px;
       color: #5c5c5c;
       text-align: center;
   }
   .question-header {
       color: #2E86C1;
   }
   .context-header {
       color: #28B463;
   }
   .answer-header {
       color: #CB4335;
   }
   .wide_text_area {
       width: 100% !important;
   }
</style>
""",
    unsafe_allow_html=True,
)

# Streamlit app!

# Create a container for the whole app
with st.container():
    # Display title and subtitle
    st.markdown("<div class='title'>Ask El</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ask our friendly English AI and get an answer!</div>",
                unsafe_allow_html=True)

    # Add an image if desired
    # st.image("path/to/your/image.jpg", width=700)

    # Question submission
    st.markdown("<h2 class='question-header'>Question</h2>",
                unsafe_allow_html=True)
    with st.container():
        st.write(
            '<style>div[data-testid="stText"] > div > textarea {width: 100% !important;}</style>',
            unsafe_allow_html=True,
        )
        question = st.text_area("", value="", height=85)

    # ... rest of the code ...


if st.button("Submit"):
    if question:
        st.write("Analyzing your question...")

        st.write("Finding the most relevant context based on your question...")
        unique_contexts = find_relevant_context(question, map, project)

        st.markdown("<h2 class='context-header'>Relevant Contexts</h2>",
                    unsafe_allow_html=True)

        # Display unique contexts
        for context in unique_contexts:
            st.write(context)

        st.write("")
        st.write("")
        st.write("El is thinking and writing...")
        relevant_context = "|||".join(unique_contexts)
        answer = ask_chatgpt(question, relevant_context)
        wrapped_answer = "\n".join(
            [textwrap.fill(p, width=80) for p in answer.split("\n")]
        )

        st.markdown("<h2 class='answer-header'>Answer</h2>",
                    unsafe_allow_html=True)
        st.write(f"{wrapped_answer}")

    else:
        st.warning("Please enter a question before submitting.")