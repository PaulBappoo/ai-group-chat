# AI Group Chat App

A Streamlit application for group chat with 4 AI participants and a human user. The AI participants are powered by different language models from OpenRouter, each with their own unique capabilities and perspectives.

## Features

- Chat with 4 different AI participants
- Configure custom system prompts for each AI
- Collaborative discussion where AIs build on each other's ideas
- Structured multi-round discussions for in-depth problem solving
- Simple setup with OpenRouter API key
- Robust error handling for model availability
- Persistent configuration storage (saves API key and prompts)
- Automatic discussion summarization at the end of each topic

## AI Participants

1. **Claude (Anthropic)** - Thoughtful and nuanced responses with detailed reasoning
2. **GPT-3.5 (OpenAI)** - Versatile and creative responses, good at explaining complex concepts
3. **Gemini (Google)** - Precise and analytical responses with methodical problem-solving
4. **Llama (Meta)** - Direct and concise responses with practical advice

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

3. Configure the application:
   - Enter your OpenRouter API key (get one from https://openrouter.ai/keys)
   - Customize the system prompts for each AI participant
   - Choose whether to save your configuration to disk
   - Click "Save and Start Chat" to begin

## Configuration Persistence

The application can save your configuration (API key and system prompts) to a local file:

- By default, settings are saved to `config.json` in the application directory
- File permissions are set to be readable only by the owner (on Unix systems)
- To disable saving, uncheck the "Save configuration to file" option in the setup screen
- Settings are automatically loaded when the application starts

## Collaborative Discussion and Summarization

When a user submits a message:

1. All AI participants provide their initial perspectives on the topic
2. The application then facilitates multiple rounds of follow-up discussion where:
   - Each AI builds on previous ideas and perspectives
   - AIs are encouraged to highlight connections between different viewpoints
   - The collective goal is to work toward a comprehensive solution
3. After several structured rounds of discussion (usually 5 rounds total), a summary is generated
4. The summary highlights key points, agreements, and the collective solution reached
5. The entire discussion remains visible, with the summary clearly demarcated

## Usage

1. Type your message or question in the input box at the bottom of the chat interface
2. Watch as the AIs engage in a collaborative discussion around your topic
3. Each AI will build upon previous responses to develop a more comprehensive solution
4. After several rounds of discussion, a summary of key points and the collective solution is generated
5. You can return to the setup screen at any time by clicking the "⚙️ Setup" button
6. If any models encounter errors, they will be displayed in an expandable section at the top of the chat

## Requirements

- Python 3.7+
- Streamlit
- OpenRouter API key 