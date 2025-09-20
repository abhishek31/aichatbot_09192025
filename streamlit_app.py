# Import necessary libraries
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel
import pickle
from utils import (
    authenticate_vertexai,
    chat_backend,
    set_background,
    replace_color,
    load_chat_history,
    update_chat_history,
    process_file,
    clear_chat_history,
    transcribe_audio,
    generate_gemini_history_object
)
import streamlit as st
import os
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
from audio_recorder_streamlit import audio_recorder


# Define variables
PROJECT_ID = "aichatbot09192025" # "walmart-rag-poc"
LOCATION = "us-central1"
CRED_FILE_PATH = 'poc_env_key.json'
METADATA_FILE_DIR = 'embeddings.pkl'
PDF_FOLDER_PATH = 'data' # folder to store uploaded files for RAG
IMAGE_SAVE_DIR = 'images' # folder to store extracted images
CHAT_HISTORY_FILE = 'chat_history.json' # file name to store chat history
BACKGROUND_IMAGE_PATH = 'background.jpg'  # chatbot background image
LOGO_IMAGE_PATH = 'logo.png'  # chatbot logo
LLM_TEMPERATURE = 1 # lower means more deterministic and higher means less deterministic
LLM_MAX_OUTPUT_TOKENS = 8192
MULTIMODAL_MODEL = "gemini-2.5-pro"
TEXT_EMBEDDING_MODEL = "text-embedding-004"
MULTIMODAL_EMBEDDING_MODEL = "multimodalembedding"
IMAGE_TEXT_EXTRACTION_TEMPERATURE = 0.3 # lower means more deterministic and higher means less deterministic
IMAGE_TEXT_EXTRACTION_MAX_OUTPUT_TOKENS = 2048
instruction = """Task: Answer the following questions in detail, providing clear reasoning and evidence from the images and text in bullet points.
    Instructions:
    1. **Analyze:** Carefully examine the provided images and text context. focusing on the process flow diagrams and the descriptions provided. Understand the sequence and purpose of each step.
    2. **Synthesize:** Integrate information from both the visual and textual elements.
    3. **Reason:**  Deduce logical connections and inferences to address the question.
    4. **Respond:** Provide a concise, accurate answer in the following format:
       * **Question:** [Question]
       * **Answer:** [Direct response to the question]
       * **Explanation:** [Bullet-point reasoning steps if applicable]
       * **Source** [name of the file, page, image from where the information is citied]
    5. **Ambiguity:** If the context is insufficient to answer, use your general knowledge to provide a helpful response. If you still cannot answer, respond "Not enough context to answer."
    """
image_description_prompt = """You are a warehouse logistics and supply chain operations expert. You will be provided with various types of images extracted from documents like warehouse standard operating procedures, warehouse product and information process flows, warehouse operational framework, warehouse organizational structure, and more.
Your task is to generate concise, accurate descriptions of the images without adding any information you are not confident about.
Focus on capturing the key details, process flows, trends, or relationships depicted in the image.

Important Guidelines:
* Prioritize accuracy:  If you are uncertain about any detail, state "Unknown" or "Not visible" instead of guessing.
* Avoid hallucinations: Do not add information that is not directly supported by the image.
* Be specific: Use precise language to describe shapes, colors, textures, and any interactions or flows depicted.
* Consider context: If the image is a screenshot or contains text, incorporate that information into your description.
"""
credentials = authenticate_vertexai(CRED_FILE_PATH, PROJECT_ID, LOCATION)

# Load LLM model from pre-trained source
multimodal_model = GenerativeModel(MULTIMODAL_MODEL)

# Load text embedding model from pre-trained source
text_embedding_model = TextEmbeddingModel.from_pretrained(TEXT_EMBEDDING_MODEL)

# Load multimodal embedding model from pre-trained source
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(MULTIMODAL_EMBEDDING_MODEL)

def main():
    set_background(BACKGROUND_IMAGE_PATH)
    img = replace_color(LOGO_IMAGE_PATH, (0,0,0), (37,39,48))
    st.logo(img)
    st.html("""
      <style>
        [alt=Logo] {
          width: 16rem;
          height: 6rem;
        }
      </style>
            """)
    if os.path.exists(METADATA_FILE_DIR):
        with open(METADATA_FILE_DIR, "rb") as f:
            data = pickle.load(f)
        text_metadata_df = data["text_metadata"] # Extract the DataFrames
        image_metadata_df = data["image_metadata"] # Extract the DataFrames
    else:
        text_metadata_df, image_metadata_df = pd.DataFrame(), pd.DataFrame()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history(CHAT_HISTORY_FILE)

    # Load multimodel model for chat
    gemini_history = generate_gemini_history_object(st.session_state.chat_history, 'User', "assistant")
    multimodal_model_llm = multimodal_model.start_chat(history = gemini_history)

    if 'current_chat_history' not in st.session_state:
        st.session_state.current_chat_history = [{"assistant": 'Hi! How can I help you?'}]

    with st.sidebar:
        st.header("Upload Documents for RAG")
        progress_bar = None
        uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
        none_state = {'break_now': False}
        none_state['break_now'] = (uploaded_files is None)
        for uploaded_file in uploaded_files:
            if not os.path.isdir(PDF_FOLDER_PATH):
                os.makedirs(PDF_FOLDER_PATH)
            text_metadata_df, image_metadata_df, progress_bar = process_file(
            uploaded_file,
            PDF_FOLDER_PATH,
            text_metadata_df,
            image_metadata_df,
            image_description_prompt,
            multimodal_model,
            text_embedding_model = text_embedding_model,
            multimodal_embedding_model = multimodal_embedding_model,
            image_save_dir = IMAGE_SAVE_DIR,
            metadat_file_dir = METADATA_FILE_DIR,
            none_chk = none_state['break_now'],
            image_text_extraction_temperature = IMAGE_TEXT_EXTRACTION_TEMPERATURE,
            image_text_extraction_max_output_tokens = IMAGE_TEXT_EXTRACTION_MAX_OUTPUT_TOKENS)
        if progress_bar is not None:
            progress_bar.empty()
        if len(text_metadata_df) > 0:
            st.caption("Uploaded documents")
            uploaded_content = text_metadata_df['file_name'].unique()
            with stylable_container(key = 'uploaded_files', css_styles="""
                {
                    background-color: #000000;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    overflow: auto;
                    white-space: nowrap;
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """):
                for item in uploaded_content:
                    if item != '.DS_Store':
                        st.write(item)

        st.header("Chat History")
        if st.session_state.chat_history:
            with stylable_container(key = 'chat_history', css_styles="""
                {
                    background-color: #000000;
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    overflow: auto;
                    white-space: nowrap;
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px)
                }
                """):
                for chat in st.session_state.chat_history:
                    btn_key = f"btn_{chat['id']}"
                    if st.button(chat['User'], key=btn_key):
                        st.session_state.selected_chat = chat

        if st.button('Clear History'):
            clear_chat_history(CHAT_HISTORY_FILE)
            st.success('Chat history cleared!')
            multimodal_model_llm = multimodal_model.start_chat()

    for msg in st.session_state.current_chat_history:
        for actor in msg:
            if actor != 'id':
                st.chat_message(actor).write(msg[actor])

    st.html(
    '''
     <style>
        iframe[title="audio_recorder_streamlit.audio_recorder"] {
            height: 2.2rem;
            position: fixed;
            bottom: 100px;
            width: 2.2rem;
            right: 388px;
            border-radius: 5px;
            padding: 1px
        }
     </style>
    '''
    )
    st.session_state.audio = audio_recorder(
            "",
            icon_size="2x",
            recording_color="#FF0000",
            neutral_color="#C0C0C0")


    prompt = st.chat_input("Type your question here")
    if 'selected_chat' in st.session_state:
        st.session_state.current_chat_history.append(st.session_state.selected_chat)
        st.chat_message("user").write(st.session_state.selected_chat['User'])
        st.chat_message("assistant").write(st.session_state.selected_chat["assistant"])
        st.session_state.pop('selected_chat', None)
    elif prompt:
            st.chat_message("user").write(prompt)
            bot_response = chat_backend(prompt, instruction, text_metadata_df, image_metadata_df, multimodal_model_llm, text_embedding_model, multimodal_embedding_model, loaded_chat_history = [st.session_state.current_chat_history, 'User', "assistant"], llm_temperature = LLM_TEMPERATURE, llm_max_output_tokens = LLM_MAX_OUTPUT_TOKENS)
            st.chat_message("assistant").write(bot_response)
            update_chat_history(prompt, bot_response, CHAT_HISTORY_FILE)
    else:
        if st.session_state.audio:
            prompt = transcribe_audio(st.session_state.audio, credentials) + " ?"
            st.chat_message("user").write(prompt)
            bot_response = chat_backend(prompt, instruction, text_metadata_df, image_metadata_df, multimodal_model_llm, text_embedding_model, multimodal_embedding_model, loaded_chat_history = [st.session_state.current_chat_history, 'User', "assistant"], llm_temperature = LLM_TEMPERATURE, llm_max_output_tokens = LLM_MAX_OUTPUT_TOKENS)
            st.chat_message("assistant").write(bot_response)
            update_chat_history(prompt, bot_response, CHAT_HISTORY_FILE)
            st.session_state.pop('audio', None)

if __name__ == "__main__":
    main()
