import streamlit as st
from rag_pipeline import create_qa_chain

@st.cache_resource
def get_qa_chain():
    return create_qa_chain()