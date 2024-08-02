# Sales_AI_Agent
This is a project that i started out to help a VC build a AI agent who communicates through audio only
The agent represents a automotive showroom agent and helps with your queries about your next car. (Watch out, it's pretty realistic and will try to sell you the pricier one!!)

## Features
- Conversation Starter: Begin a conversation with the AI assistant.
- Audio to Text: Transcribe audio to text using an OpenAI model.
- Fast Inference: Generate responses quickly with Groq.
- SQL Agent: Generates SQL query to query the PostgreSQL instance to retrieve top_k records ordered by price, desc. Then create the response
- Text to Speech: Convert text back to speech using gTTS.
- Frontend Design: Design the frontend interface using Streamlit.

## Features
This is just to showcase what I could pull together in a few weaks. There is a cleaner way to build this with Langgraph and Im learning the best practices

## Installation

1. Clone the repository:
    ```bash
    git clone 
   
2. Navigate to the project directory:
    ```bash
    cd ai-voice-assistant
    ```
3. Create and activate virtual environment:
    ```bash
    python -m venv venv
    venv/Scripts/activate
    ```
4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Create a `.env` file using `.env-example` as a template:
    ```bash
    cp .env-example .env
     ```
2. Run the main application script:
    ```bash
    streamlit run src/app.py
    ```
## Actively looking out for  a job
Hire me!!

