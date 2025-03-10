import streamlit as st
import requests
import json
import os
from datetime import datetime
from pathlib import Path
from document_processor import (
    extract_text, process_document, get_embedding_model,
    setup_chroma, generate_embeddings, store_chunks, 
    retrieve_relevant_chunks, simple_search
)

# Set page configuration
st.set_page_config(page_title="AI Group Chat", layout="wide")

# Path for configuration file
CONFIG_PATH = Path("config.json")

# Define AI participants
AI_PARTICIPANTS = {
    "Claude": {
        "model": "anthropic/claude-3-haiku",  # Changed to a more widely available Claude model
        "color": "#9575CD",  # Light purple
    },
    "GPT-4o": {  # Changed from GPT-3.5 to GPT-4o
        "model": "openai/gpt-4o",
        "color": "#4CAF50",  # Green
    },
    "Gemini": {  # Changed from Mistral to Gemini
        "model": "google/gemini-pro",
        "color": "#2196F3",  # Blue
    },
    "Llama": {
        "model": "meta-llama/llama-3-8b-instruct",  # Changed to the smaller Llama model
        "color": "#FF9800",  # Orange
    }
}

# Define default system prompts
DEFAULT_SYSTEM_PROMPTS = {
    "Claude": "You are Claude, a thoughtful and nuanced AI assistant in a group chat with other AIs and a human. You provide detailed and well-reasoned responses. You should actively engage with ideas from other AIs, building on them or offering alternative perspectives when appropriate. If you don't have anything meaningful to contribute, simply respond with 'SKIP_RESPONSE' and your turn will be skipped.",
    "GPT-4o": "You are GPT-4o, a versatile and creative AI assistant in a group chat with other AIs and a human. You excel at generating ideas and explaining complex concepts simply. You should actively engage with ideas from other AIs, building on them or offering alternative perspectives when appropriate. If you don't have anything meaningful to contribute, simply respond with 'SKIP_RESPONSE' and your turn will be skipped.",
    "Gemini": "You are Gemini, a precise and analytical AI assistant in a group chat with other AIs and a human. You're good at solving problems methodically. You should actively engage with ideas from other AIs, building on them or offering alternative perspectives when appropriate. If you don't have anything meaningful to contribute, simply respond with 'SKIP_RESPONSE' and your turn will be skipped.",
    "Llama": "You are Llama, a direct and concise AI assistant in a group chat with other AIs and a human. You provide straightforward, practical advice. You should actively engage with ideas from other AIs, building on them or offering alternative perspectives when appropriate. If you don't have anything meaningful to contribute, simply respond with 'SKIP_RESPONSE' and your turn will be skipped."
}

def save_config():
    """Save the current configuration to a file."""
    config = {
        "openrouter_api_key": st.session_state.openrouter_api_key,
        "system_prompts": st.session_state.system_prompts,
        "max_rounds_per_prompt": st.session_state.max_rounds_per_prompt
    }
    
    # Create a secure file with restricted permissions
    try:
        # Write to the file
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        
        # Set file permissions to be readable only by the owner (unix systems)
        if os.name == "posix":  # Unix-like systems
            os.chmod(CONFIG_PATH, 0o600)
        
        return True
    except Exception as e:
        st.error(f"Failed to save configuration: {str(e)}")
        return False

def load_config():
    """Load configuration from file if it exists."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            
            # Update session state with loaded values
            st.session_state.openrouter_api_key = config.get("openrouter_api_key", "")
            st.session_state.system_prompts = config.get("system_prompts", {})
            st.session_state.max_rounds_per_prompt = config.get("max_rounds_per_prompt", 5)
            
            # If system_prompts doesn't contain all AI participants, initialize the missing ones
            for ai_name in AI_PARTICIPANTS.keys():
                if ai_name not in st.session_state.system_prompts:
                    st.session_state.system_prompts[ai_name] = DEFAULT_SYSTEM_PROMPTS[ai_name]
            
            return True
        except Exception as e:
            st.error(f"Failed to load configuration: {str(e)}")
            return False
    
    return False

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = ""
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'system_prompts' not in st.session_state:
    st.session_state.system_prompts = DEFAULT_SYSTEM_PROMPTS.copy()
if 'last_speaking_ai' not in st.session_state:
    st.session_state.last_speaking_ai = None
if 'active_discussion' not in st.session_state:
    st.session_state.active_discussion = False
if 'discussion_counter' not in st.session_state:
    st.session_state.discussion_counter = 0
if 'model_errors' not in st.session_state:
    st.session_state.model_errors = {}
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = False
if 'waiting_for_summary' not in st.session_state:
    st.session_state.waiting_for_summary = False
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False
if 'discussion_messages' not in st.session_state:
    st.session_state.discussion_messages = []
if 'max_rounds_per_prompt' not in st.session_state:
    st.session_state.max_rounds_per_prompt = 5  # Default to 5 rounds
if 'has_document' not in st.session_state:
    st.session_state.has_document = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = None

def setup_view():
    """Display the setup view for configuring the OpenRouter API key and system prompts."""
    st.title("AI Group Chat Setup")
    
    # API Key input
    api_key = st.text_input("OpenRouter API Key", 
                           value=st.session_state.openrouter_api_key,
                           type="password", 
                           help="You can get this from https://openrouter.ai/keys")
    
    # Max Rounds Per Prompt input
    max_rounds = st.number_input(
        "Max Rounds Per Prompt",
        min_value=1,
        max_value=10,
        value=st.session_state.max_rounds_per_prompt,
        step=1,
        help="Maximum number of rounds of discussion before generating a summary. Each AI will respond once per round."
    )
    
    st.subheader("System Prompts for AI Participants")
    st.write("Configure how each AI participant should behave in the conversation")
    
    # System prompt inputs for each AI participant
    system_prompts = {}
    for ai_name in AI_PARTICIPANTS.keys():
        system_prompts[ai_name] = st.text_area(
            f"System Prompt for {ai_name}",
            value=st.session_state.system_prompts.get(ai_name, DEFAULT_SYSTEM_PROMPTS.get(ai_name, "")),
            height=100
        )
    
    # Save options
    save_config_option = st.checkbox("Save configuration to file (includes API key)", value=True)
    
    # Save button
    if st.button("Save and Start Chat"):
        st.session_state.openrouter_api_key = api_key
        st.session_state.system_prompts = system_prompts
        st.session_state.max_rounds_per_prompt = max_rounds
        
        # Save configuration if requested
        if save_config_option:
            if save_config():
                st.success("Configuration saved successfully!")
            else:
                st.warning("Failed to save configuration, but you can continue with the current session.")
        
        st.session_state.setup_complete = True
        st.rerun()

def call_openrouter_api(messages, model, max_tokens=1000):
    """Call the OpenRouter API with the given messages and model."""
    if not st.session_state.openrouter_api_key:
        st.error("OpenRouter API key is missing. Please go to setup.")
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.session_state.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8501",  # Added referer which can be needed
        "X-Title": "AI Group Chat"  # Added title
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

    try:
        st.sidebar.write(f"Calling OpenRouter API for model: {model}")
        response = requests.post(url, headers=headers, json=data)
        
        # Debug response status
        st.sidebar.write(f"Response status code: {response.status_code}")
        
        # Check for error response
        if response.status_code != 200:
            error_message = f"API Error: Status {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message += f" - {error_data['error']['message']}"
            except:
                error_message += f" - {response.text[:100]}"
                
            raise requests.exceptions.RequestException(error_message)
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        st.sidebar.write(error_msg)
        return None

def get_ai_response(ai_name, message_history, is_summary=False):
    """Get a response from the specified AI participant."""
    model = AI_PARTICIPANTS[ai_name]["model"]
    system_prompt = st.session_state.system_prompts[ai_name]
    
    # Clear any previous errors for this model
    if ai_name in st.session_state.model_errors:
        del st.session_state.model_errors[ai_name]
    
    # Prepare special instructions based on whether this is a normal response or summary
    if is_summary:
        additional_context = """
        You are asked to create a summary of the discussion that just took place.
        1. Summarize the key points from the discussion
        2. Highlight any agreements or consensus reached
        3. Note any important disagreements or open questions
        4. Keep the summary concise but comprehensive, around 150-250 words
        5. Start your summary with "## Discussion Summary:"
        6. End with the key conclusion or solution that emerged from the collaborative discussion
        """
    else:
        additional_context = """
        Collaborative Group Chat Rules:
        1. Actively build on ideas from other participants
        2. Offer unique insights or perspectives
        3. If others have already covered a point sufficiently, don't repeat it
        4. Work toward a collective solution or consensus
        5. Highlight connections between different viewpoints
        6. If you truly don't have a meaningful contribution, respond with 'SKIP_RESPONSE'
        7. Consider how your expertise complements what others have said
        8. If appropriate, synthesize or integrate previous points into a more cohesive view
        
        If the user has provided document context, make sure to use that information in your response.
        """
    
    combined_prompt = f"{system_prompt}\n\n{additional_context}"
    
    # Prepare messages for API call
    messages = [{"role": "system", "content": combined_prompt}]
    
    # Use the current_query (with document context) for the first AI response
    if len(message_history) > 0 and message_history[-1]["role"] == "user" and hasattr(st.session_state, 'current_query'):
        # Replace the last user message with the enhanced version containing document context
        for i in range(len(message_history) - 1):
            msg = message_history[i]
            role = "assistant" if msg["role"] in AI_PARTICIPANTS else msg["role"]
            content = msg["content"]
            # Identify who is speaking for context
            if role == "assistant":
                content = f"[{msg['role']}]: {content}"
            messages.append({"role": role, "content": content})
        
        # Add the enhanced message with document context
        messages.append({"role": "user", "content": st.session_state.current_query})
    else:
        # Add message history normally
        for msg in message_history:
            role = "assistant" if msg["role"] in AI_PARTICIPANTS else msg["role"]
            content = msg["content"]
            # Identify who is speaking for context
            if role == "assistant":
                content = f"[{msg['role']}]: {content}"
            messages.append({"role": role, "content": content})
    
    # Add a special instruction based on the response type
    if is_summary:
        messages.append({
            "role": "user", 
            "content": f"Please provide a summary of the discussion above as {ai_name}. Include key points, agreements, and any collective solutions reached."
        })
    else:
        # Different instruction based on whether this is the first round or a follow-up
        first_round = all(msg["role"] == "user" for msg in message_history[-1:])
        if first_round:
            messages.append({
                "role": "user", 
                "content": f"It's now your turn as {ai_name} to respond to the human's query. Provide your initial thoughts and insights on the topic."
            })
        else:
            # If this is a follow-up round, actively encourage building on others' ideas
            messages.append({
                "role": "user", 
                "content": f"It's now your turn as {ai_name} to continue the discussion. Consider the perspectives already shared and build upon them, add new insights, or help synthesize toward a collective solution. If you don't have anything meaningful to add, respond with 'SKIP_RESPONSE'."
            })
    
    # Call API
    with st.spinner(f"{ai_name} is {'summarizing the discussion' if is_summary else 'thinking'}..."):
        try:
            response = call_openrouter_api(messages, model, max_tokens=1500 if is_summary else 1000)
        except Exception as e:
            st.sidebar.write(f"Error with {ai_name}: {str(e)}")
            st.session_state.model_errors[ai_name] = str(e)
            return None
        
    if response and "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        # Check if the AI decided to skip (only for regular responses, not summaries)
        if not is_summary and "SKIP_RESPONSE" in content:
            return None
        return content
    else:
        # Store error if response is empty or invalid
        error_msg = "No valid response received"
        st.session_state.model_errors[ai_name] = error_msg
        st.sidebar.write(f"Error with {ai_name}: {error_msg}")
        return None

def chat_view():
    """Display the chat interface."""
    st.title("AI Group Chat")
    
    # Set up columns for document upload and chat
    col1, col2 = st.columns([1, 3])
    
    # Initialize document storage in session state if not present
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
        
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    
    with col1:
        # Document upload section
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Upload document for discussion", type=["txt", "pdf"])
        
        if uploaded_file:
            try:
                # Only process if it's a new document
                if st.session_state.doc_name != uploaded_file.name:
                    with st.spinner("Processing document..."):
                        # Extract text from uploaded file
                        text = extract_text(uploaded_file)
                        
                        # Process document into chunks
                        chunks = process_document(text)
                        
                        # Store in session state directly (no embeddings)
                        st.session_state.doc_chunks = chunks
                        st.session_state.doc_name = uploaded_file.name
                        st.session_state.has_document = True
                        
                        st.success(f"Document processed: {len(chunks)} chunks created")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
        
        # Show document status
        if st.session_state.get('has_document', False):
            st.info(f"Active document: {st.session_state.get('doc_name', 'Document')}")
            if st.button("Clear Document"):
                st.session_state.has_document = False
                st.session_state.doc_chunks = []
                st.session_state.doc_name = None
                st.rerun()
    
    with col2:
        # Display chat header
        st.subheader("AI Group Chat")
        
        # Display setup button to return to setup view
        if st.button("⚙️ Setup"):
            st.session_state.setup_complete = False
            st.rerun()
        
        # Add a reset button to clear the chat history
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.session_state.message_history = []
            st.session_state.active_discussion = False
            st.session_state.discussion_counter = 0
            st.session_state.discussion_messages = []
            st.session_state.waiting_for_summary = False
            st.session_state.summary_generated = False
            st.session_state.last_speaking_ai = None
            st.session_state.model_errors = {}
            st.session_state.current_query = None
            st.rerun()
        
        # Display any model errors in a collapsible section
        if st.session_state.model_errors:
            with st.expander("Model Errors (Click to expand)"):
                st.error("Some models encountered errors in the last request:")
                for model, error in st.session_state.model_errors.items():
                    st.write(f"**{model}**: {error}")
                if st.button("Clear Errors"):
                    st.session_state.model_errors = {}
                    st.rerun()
        
        # Display the chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.chat_message("user", avatar="👤").write(message["content"])
                elif message["role"] == "summary":
                    # Display summary in a distinctive way
                    st.markdown("---")
                    st.markdown(message["content"], unsafe_allow_html=True)
                    st.markdown("---")
                else:
                    ai_name = message["role"]
                    color = AI_PARTICIPANTS[ai_name]["color"]
                    st.chat_message(ai_name, avatar=f"🤖").markdown(
                        f"<span style='color:{color}'><strong>{ai_name}:</strong> {message['content']}</span>", 
                        unsafe_allow_html=True
                    )
    
        # Input for user message
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Reset discussion tracking when a new user message is received
            st.session_state.active_discussion = False
            st.session_state.waiting_for_summary = False
            st.session_state.summary_generated = False
            st.session_state.discussion_counter = 0
            st.session_state.discussion_messages = []
            st.session_state.last_speaking_ai = None
            
            # Process user input with document context if available
            if st.session_state.get('has_document', False) and st.session_state.doc_chunks:
                # Use simple search to find relevant chunks
                relevant_chunks = simple_search(user_input, st.session_state.doc_chunks)
                
                # Format context message with retrieved chunks
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    enhanced_input = f"User question: {user_input}\n\nRelevant document context:\n{context}\n\nPlease use the document context to help answer the question."
                    
                    # Store original query for display, but use enhanced input for processing
                    display_message = user_input
                    process_message = enhanced_input
                else:
                    display_message = user_input
                    process_message = user_input
            else:
                display_message = user_input
                process_message = user_input
            
            # Add user message to chat history for display
            st.session_state.messages.append({"role": "user", "content": display_message})
            
            # Store the processed message with context for AI processing
            st.session_state.current_query = process_message
            
            # Rerun to display the new message immediately
            st.rerun()

def check_for_disagreement(recent_messages):
    """Check if there appears to be a disagreement in the recent messages."""
    disagreement_keywords = [
        "disagree", "however", "but", "on the contrary", "I think", 
        "different view", "not necessarily", "alternative", "instead", 
        "rather", "contrary", "oppose", "differ", "conflict", "dispute"
    ]
    
    for keyword in disagreement_keywords:
        if keyword in recent_messages.lower():
            return True
    
    return False

def check_for_conclusion(recent_messages):
    """Check if the discussion seems to have reached a conclusion."""
    conclusion_keywords = [
        "in conclusion", "to summarize", "therefore", "thus", "in summary",
        "we agree", "consensus", "resolved", "settled", "agreed", "compromise",
        "common ground", "solution", "resolution"
    ]
    
    for keyword in conclusion_keywords:
        if keyword in recent_messages.lower():
            return True
    
    return False

def generate_summary():
    """Generate a summary of the discussion."""
    # Choose the AI that will generate the summary (using Claude for better summarization)
    summarizer_ai = "Claude"
    
    # Get the summary
    summary = get_ai_response(summarizer_ai, st.session_state.discussion_messages, is_summary=True)
    
    if summary:
        # Add the summary to the chat history
        st.session_state.messages.append({"role": "summary", "content": summary})
        st.session_state.summary_generated = True
    
    st.session_state.waiting_for_summary = False

def generate_ai_responses():
    """Generate responses from AI participants if needed."""
    if not st.session_state.messages or st.session_state.active_discussion:
        return
    
    # Debug information
    st.sidebar.write("Debug Information:")
    st.sidebar.write(f"Discussion counter: {st.session_state.discussion_counter}")
    st.sidebar.write(f"Active discussion: {st.session_state.active_discussion}")
    st.sidebar.write(f"Waiting for summary: {st.session_state.waiting_for_summary}")
    st.sidebar.write(f"Summary generated: {st.session_state.summary_generated}")
    st.sidebar.write(f"Last speaking AI: {st.session_state.last_speaking_ai}")
    
    # Check if we need to generate a summary
    if st.session_state.waiting_for_summary and not st.session_state.summary_generated:
        generate_summary()
        st.rerun()
        return
    
    # Get the last message
    last_message = st.session_state.messages[-1]
    
    # If the last message is from a user, start a new discussion
    if last_message["role"] == "user":
        st.session_state.active_discussion = True
        message_history = st.session_state.messages.copy()
        successful_responses = 0
        
        # Store all messages in this discussion for the summary
        st.session_state.discussion_messages = message_history.copy()
        
        # Let each AI participant respond with their initial thoughts
        for ai_name in AI_PARTICIPANTS.keys():
            response = get_ai_response(ai_name, message_history)
            if response:
                # Add AI response to chat history
                st.session_state.messages.append({"role": ai_name, "content": response})
                message_history.append({"role": ai_name, "content": response})
                st.session_state.discussion_messages.append({"role": ai_name, "content": response})
                st.session_state.last_speaking_ai = ai_name
                successful_responses += 1
        
        # After initial responses, always initiate a follow-up discussion round
        # to encourage collaborative problem-solving
        st.session_state.active_discussion = False
        st.session_state.discussion_counter = 1  # We've completed the first round
        
        # If we got responses, continue to the collaborative discussion phase
        if successful_responses > 0:
            st.rerun()
    
    # Collaborative discussion phase - multiple rounds of follow-ups
    elif len(st.session_state.messages) >= 2 and st.session_state.discussion_counter < st.session_state.max_rounds_per_prompt:
        # Continue discussion until we reach the max_rounds_per_prompt limit
        st.session_state.active_discussion = True
        
        # Select the next AI to build on the discussion
        if st.session_state.last_speaking_ai:
            current_ai_index = list(AI_PARTICIPANTS.keys()).index(st.session_state.last_speaking_ai)
            next_ai_index = (current_ai_index + 1) % len(AI_PARTICIPANTS)
            next_ai = list(AI_PARTICIPANTS.keys())[next_ai_index]
        else:
            next_ai = list(AI_PARTICIPANTS.keys())[0]
        
        # Add debug information
        st.sidebar.write(f"Next AI to speak: {next_ai}")
        
        # Skip models that had errors
        if next_ai in st.session_state.model_errors:
            st.sidebar.write(f"Skipping {next_ai} due to errors")
            st.session_state.active_discussion = False
            st.session_state.discussion_counter += 1  # Count it as a round even if skipped
            st.rerun()
            return
        
        # Get the follow-up response that builds on previous messages
        response = get_ai_response(next_ai, st.session_state.messages)
        
        if response:
            # Add to messages and discussion tracking
            st.session_state.messages.append({"role": next_ai, "content": response})
            st.session_state.discussion_messages.append({"role": next_ai, "content": response})
            st.session_state.last_speaking_ai = next_ai
            st.session_state.discussion_counter += 1
            st.sidebar.write(f"Response received from {next_ai}")
        else:
            st.sidebar.write(f"No response received from {next_ai}")
        
        # Always go to the next round until we reach the limit
        st.session_state.active_discussion = False
        
        # Check if we've reached the end of the discussion rounds
        if st.session_state.discussion_counter >= st.session_state.max_rounds_per_prompt:
            st.session_state.waiting_for_summary = True
        
        st.rerun()
    
    # After all discussion rounds, generate a summary
    elif st.session_state.discussion_counter >= st.session_state.max_rounds_per_prompt and not st.session_state.waiting_for_summary:
        st.session_state.waiting_for_summary = True
        st.rerun()

def main():
    """Main entry point of the application"""
    # Initialize session state variables
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = ""
        
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
        
    if "active_discussion" not in st.session_state:
        st.session_state.active_discussion = False
        
    if "discussion_counter" not in st.session_state:
        st.session_state.discussion_counter = 0
        
    if "discussion_messages" not in st.session_state:
        st.session_state.discussion_messages = []
        
    if "waiting_for_summary" not in st.session_state:
        st.session_state.waiting_for_summary = False
        
    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False
        
    if "last_speaking_ai" not in st.session_state:
        st.session_state.last_speaking_ai = None
        
    if "model_errors" not in st.session_state:
        st.session_state.model_errors = {}
        
    if "max_rounds_per_prompt" not in st.session_state:
        st.session_state.max_rounds_per_prompt = 5
        
    if "config_loaded" not in st.session_state:
        st.session_state.config_loaded = False
        
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
        
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = DEFAULT_SYSTEM_PROMPTS.copy()
        
    if "has_document" not in st.session_state:
        st.session_state.has_document = False
        
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = []
        
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None

    # Load configuration on first run
    if not st.session_state.config_loaded:
        load_config()
        st.session_state.config_loaded = True
        
        # If we loaded a valid API key, mark setup as complete
        if st.session_state.openrouter_api_key:
            st.session_state.setup_complete = True
    
    if not st.session_state.setup_complete:
        setup_view()
    else:
        chat_view()
        try:
            generate_ai_responses()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.sidebar.write(f"Exception in generate_ai_responses: {str(e)}")
            import traceback
            st.sidebar.write(traceback.format_exc())

if __name__ == "__main__":
    main() 