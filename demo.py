import gradio as gr
import random
import time

# Preloaded answers
preloaded_answers = [
    "Would you like to translate the sign language video?",
    "Do you need assistance with specific signs?",
    "Do you want to save this video for later use?",
    "Would you like to upload another video?"
]


# Define a function to handle video input and generate responses
def handle_video(video, chat_history):
    response = "Video received. What would you like to do next?"
    chat_history.append((None, "Try to sign 'team', please!"))
    chat_history.append(("(Video Posted)", None))
    chat_history.append((None, "You should position your hands in front of her chest."))
    chat_history.append((None, "You should change the handshape of your hands to be in a fist."))
    return chat_history


# Create the Gradio interface
with gr.Blocks(css=".btn {background-color: #5AC4F6;}", theme=gr.themes.Soft(font=["-apple-system", "BlinkMacSystemFont", "sans-serif"])) as demo:
    gr.Markdown(
        """
        # LLM Providing Feedback to Sign Language Learners

        **Instructions**:
        1. Upload a video of someone performing sign language on the left.
        2. Use the chat interface on the right to interact and get suggestions on how to proceed.
        3. Click 'Submit Video' after uploading your video.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Sign Language Video")
            submit_button = gr.Button("Submit Video", elem_id="submit-button")
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat Interface", height=500)
            msg = gr.Textbox(label="Follow-ups", placeholder="Type a message...")

    # Button to handle video submission
    submit_button.click(handle_video, inputs=[video_input, chatbot], outputs=[chatbot])


    # Preload answers into the chat interface
    def preload_answers(chat_history):
        for answer in preloaded_answers:
            chat_history.append(("Bot", answer))
        return chat_history


    demo.load(preload_answers, inputs=None, outputs=[chatbot])

# Apply CSS for styling
demo.css += """
.btn {
    background-color: #5AC4F6;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s;
}

.btn:hover {
    background-color: #4AA3D6;
}

.gr-chatbot {
    border: 2px solid #5AC4F6;
    border-radius: 5px;
    padding: 10px;
}

footer {
    opacity: 0;
}

"""

# Launch the app
demo.launch()
