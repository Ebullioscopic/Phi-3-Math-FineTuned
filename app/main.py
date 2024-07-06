# phi3_chat_interface.py
from st_chat_message import message
import streamlit as st
from streamlit.components.v1 import html
from huggingface_hub import login
import dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
dotenv.load_dotenv()

# Hugging Face login token
#hutoken = os.getenv('HF_TOKEN')
hutoken = "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"
login(token=hutoken)

# Define the chat system prompt
chat_system_prompt = """
    You are an advanced assistant that always greets users with catchy taglines for making them solve mathematical word problems.
    You can help them solve problems based on their requirements, budget etc.
    While solving any problem, give the response in a plain text format.
    If the user tries to ask out of topic questions do not engage in the conversation.
    If the given context is not sufficient to solve the problem, do not answer the question.
"""

@st.cache_resource
def llm():
    # Loading Model and Tokenizer for phi_3
    model_checkpoint = 'Ebullioscopic/phi_3_math_full_fine_tuned'
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True, device_map='auto', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)

    # Pipeline Creation
    pipy = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, max_length=3000, device_map="auto")
    llmpipe = HuggingFacePipeline(pipeline=pipy)
    return llmpipe

@st.cache_resource
def retriever():
    # Loading the math dataset
    dataset_loader = HuggingFaceDatasetLoader("Ebullioscopic/orca-math-word-problems-200k")
    texts = dataset_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(texts)

    # Creating the vectorstore
    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(collection_name="sample_collection", embedding_function=embedding_function)
    vectorstore.add_documents(texts)

    # Creating the retriever
    retrieverdata = vectorstore.as_retriever(k=7)
    return retrieverdata

class Pipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def retrieve(self, problem):
        docs = self.retriever.invoke(problem)
        return "\n\n".join([d.page_content for d in docs])

    def augment(self, problem):
        return f"""
        system
        {chat_system_prompt}
        user
        Solve the problem based on the context provided below
        Context: {self.retrieve(problem)}
        Problem: {problem}
        assistant
        """

    def parse(self, string):
        return string

    def generate(self, problem):
        prompt = self.augment(problem)
        answer = self.llm.invoke(prompt)
        return self.parse(answer)

@st.cache_resource    
def pipe():    
    piped = Pipeline(llm(), retriever())
    return piped

def chat(prompt):
    if prompt is not None:
        output = pipe().generate(prompt)
        return output

def on_input_change():
    user_input = st.session_state.user_input
    if len(user_input) == 0:
        return None
    st.session_state.msgs.append({"content": str(user_input), "is_user": True})
    value = chat(user_input)
    st.session_state.msgs.append({"content": str(value), "is_user": False})

if "msgs" not in st.session_state:
    st.session_state.msgs = [
        {
            "content": "Hi, I am MathAI, your virtual assistant. Ask me any mathematical word problem. I will try to solve it. Thank you.",
            "is_user": False,
        },
    ]

chat_placeholder = st.empty()

with chat_placeholder.container():
    for idx, msg in enumerate(st.session_state.msgs):
        message(msg["content"], is_user=msg["is_user"], key=f"message_{idx}")

with st.container():
    st.text_input("Problem:", on_change=on_input_change, key="user_input")

