import streamlit as st
import io
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document processing functions
def extract_text(uploaded_file):
    """Extract text from uploaded file based on file type"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return extract_from_pdf(uploaded_file)
    elif file_type == 'txt':
        return uploaded_file.getvalue().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def extract_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def estimate_tokens(text):
    """Roughly estimate token count based on word count (simple approximation)"""
    return len(text.split()) * 1.3  # Rough estimation: 1.3 tokens per word on average

def process_document(text):
    """Split document into chunks with approximate token count awareness"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=lambda x: len(x.split()) * 1.3,  # Simple word-based approximation
    )
    chunks = splitter.split_text(text)
    return chunks

# Simple keyword-based search for chunks
def simple_search(query, chunks, k=3):
    """Basic keyword search for relevant chunks"""
    # Convert query to lowercase for case-insensitive matching
    query_terms = query.lower().split()
    
    # Score each chunk based on keyword matches
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for term in query_terms if term in chunk_lower)
        chunk_scores.append((i, score))
    
    # Sort by score (descending) and get top k
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:k]
    
    # Return the actual chunks
    return [chunks[i] for i, _ in sorted_chunks if _ > 0]

# Placeholder functions to maintain compatibility with app.py
def get_embedding_model():
    """Placeholder function to maintain API compatibility"""
    return None

def setup_chroma():
    """Placeholder function to maintain API compatibility"""
    return None

def generate_embeddings(chunks, model):
    """Placeholder function to maintain API compatibility"""
    return chunks

def store_chunks(collection, chunks, embeddings):
    """Placeholder function to maintain API compatibility"""
    return chunks

def retrieve_relevant_chunks(query, collection, embedding_model, k=3):
    """Replaced with simple keyword search"""
    if not isinstance(collection, list):
        # If we're getting None from setup_chroma
        return []
    
    return simple_search(query, collection, k) 