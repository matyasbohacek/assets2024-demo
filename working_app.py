import gradio as gr
from PIL import Image
import sys
import cv2
import numpy as np
import random
from io import BytesIO
import torch
from working_utils import chat_with_model
from working_utils import load_image
from working_utils import parse_args
from working_utils import keyframe_extraction

# EXAMPLES should now only hold reference images and videos
examples = [
    ("./examples/bad.png", "./examples/bad.mp4"),
    ("./examples/great.png", "./examples/great.mp4"),
    ("./examples/mid.png", "./examples/mid.mp4"),
]

EXAMPLES = {
    "bad": examples[0],
    "great": examples[1],
    "mid": examples[2]
}

def generate_html_analysis(overall_assessment, reasoning_handshape, reasoning_movement,
                           image_path_1, image_path_2):
    # Determine the class and symbol based on the overall assessment
    if "relevant" in overall_assessment:
        assessment_class = "KEEP IT UP!"
        check_handshape = True
        check_movement = True
        symbol = "✓"
        color = "#78a498"
    elif "need" in overall_assessment and "improvement" in overall_assessment:
        assessment_class = "needs-improvement"
        check_handshape = False
        check_movement = True
        symbol = "!"
        color = "#ff7f0e"
    else:  # "MORE WORK TO DO"
        assessment_class = "not-relevant"
        check_handshape = False
        check_movement = False
        symbol = "✗"
        color = "#EE4266"

    # Create HTML structure
    html_output = f"""
    <div class="container-spec">
        <div id="overallAssessment" class="assessment {overall_assessment}" style="background-color: {color};">
            <div style='color: #fff !important; font-weight: bold;'>{assessment_class}</div><div class="check-mark" style='color: #fff !important;'>{symbol}</div>
        </div>
        <div class="reasoning">
            <strong>Handshape:</strong> <span>{"LOOKS GOOD" if check_handshape else "LET'S IMPROVE THIS"} <br><br> {reasoning_handshape}</span>
        </div>
        <div class="reasoning">
            <strong>Hand Movement:</strong> <span>{"LOOKS GOOD" if check_movement else "LET'S IMPROVE THIS"} <br><br> {reasoning_movement}</span>
        </div>
        <div class="image-container" style="display: flex; justify-content: space-between;">
            <img src="{image_path_1}" alt="Image 1" style="width: 48%; border-radius: 10px;">
            <img src="{image_path_2}" alt="Image 2" style="width: 48%; border-radius: 10px;">
        </div>
    </div>
    """

    return html_output



def merge_images(image1, image2):
    widths, heights = zip(*(i.size for i in [image1, image2]))
    total_width = sum(widths)
    max_height = max(heights)

    merged_image = Image.new('RGB', (total_width, max_height))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (image1.width, 0))

    return merged_image

def handle_video(video, sign_choice, chat_history):
    reference_image_url, reference_video_path = EXAMPLES[sign_choice]

    args = parse_args()
    i3d_checkpoint_path = "sign_utils/i3d/i3d_kinetics_bslcp.pth.tar"
    mstcn_checkpoint_path = "sign_utils/ms-tcn/mstcn_bslcp_i3d_bslcp.model"
    save_path = "./tmp/highest_prob.png"
    os.makedirs("./tmp", exist_ok=True)

    # Extract a key frame from the uploaded video
    user_key_frame_path = keyframe_extraction(
        video, i3d_checkpoint_path, mstcn_checkpoint_path, save_path
    )

    # Load the reference image and user's key frame
    reference_image = load_image(reference_image_url)
    user_key_frame = load_image(user_key_frame_path)

    # Debug: Check image types
    print(f"user_key_frame type: {type(user_key_frame)}")
    print(f"reference_image type: {type(reference_image)}")

    # Merge the images side by side
    merged_image = merge_images(user_key_frame, reference_image)
    merged_image_path = "./tmp/merged_image.png"

    try:
        merged_image.save(merged_image_path)
    except Exception as e:
        print(f"Error saving merged image: {e}")

    # Call the chat_with_model function
    overall_assessment, _ = chat_with_model(merged_image_path, "ONLY reply with one of these words: relevant, needs-improvement, or not-good! How similar are the sign movements in these two images? NO other text!", args)
    print(f"Overall assessment: {overall_assessment}")
    hand_shape, _ = chat_with_model(merged_image_path, "In one sentence, what gesture/hand shape changes should the person on the left make to match the person on the right?", args)
    print(f"Hand shape: {hand_shape}")
    hand_movement, _ = chat_with_model(merged_image_path, "In one sentence, how should the person on the left position her hands to look like the person on the right", args)
    print(f"Hand movement: {hand_movement}")
    # Generate the HTML content
    html_content = generate_html_analysis(
        overall_assessment,  # LLM generated assessment 
        hand_shape,  
        hand_movement,  
        merged_image_path,  # First image path (merged image)
        reference_image_url  # Second image path (reference image)
    )

    chat_history.clear()  # Reset chat history as per the original requirement

    #return html_content, chat_history, gr.update(visible=True), gr.update(visible=True, value=reference_video_path), gr.update(visible=True)
    return html_content, chat_history
def handle_chat_input(chat_input, chat_history):

    args = parse_args()
    # Placeholder: Simulate a conversation response using the model
    response, _ = chat_with_model("/tmp/merged_image.png", chat_input, args)
    chat_history.append((chat_input, response))
    return chat_history

with gr.Blocks(css=".btn {background-color: #357edd;}",
               theme=gr.themes.Soft(font=["-apple-system", "BlinkMacSystemFont", "sans-serif"])) as demo:
    gr.Markdown("<br>")
    gr.Markdown("# Sign Language Feedback")
    gr.Markdown("Choose a sign you want to perfect and sign it on the camera!")

    with gr.Row():
        with gr.Column():
            sign_choice = gr.Dropdown(label="Choose a Sign to Perform", choices=["bad", "great", "mid"])
            video_input = gr.Video(label="Upload Sign Language Video")
            submit_button = gr.Button("Submit Video", elem_id="submit-button")
        with gr.Column():
            output_html = gr.HTML(visible=True)  # HTML visible from the start
            with gr.Accordion("Reference", open=False, visible=False) as reference_vid_wr:
                reference_video = gr.Video(label="Reference Video", visible=False)  # Hidden reference video
            with gr.Accordion("Follow-up Chat", open=False, visible=False) as chat_module:
                chatbot = gr.Chatbot(label="Chat Interface", height=300)  # Reduced height for smaller vertical size
                with gr.Row():
                    with gr.Column(scale=4):
                        chat_input = gr.Textbox(placeholder="Type your message here...", show_label=False)
                    with gr.Column(scale=1):
                        chat_submit = gr.Button("Send", elem_classes="chat-button")

    # Integrate chat_with_model into the submit_button click function
    submit_button.click(handle_video, inputs=[video_input, sign_choice, chatbot],
                        outputs=[output_html, chatbot])

    chat_submit.click(handle_chat_input, inputs=[chat_input, chatbot], outputs=[chatbot])

demo.css += """
html {
    background-color: #F5F5F2;
}

input.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j:checked, input.svelte-1mhtq7j.svelte-1mhtq7j.svelte-1mhtq7j:checked {
    background-color: #357edd;
    border-bottom-color: #357edd;
    border-left-color: #357edd;
    border-right-color: #357edd;
    border-top-color: #357edd;
    color: #357edd;
}

.svelte-1gfkn6j {
    font-weight: bold !important;
}

gradio-app {
    background:  #F5F5F2 !important;
    background-color:  #F5F5F2 !important;
}

body, .gradio-container {
    background-color: #F5F5F2; /* Light mode background */
    color: #333333; /* Dark text color */
    font-family: Helvetica, Arial, sans-serif; /* Set font to Helvetica */
    font-size: 18px; /* Increase text size */
}

.built-with {
    display: none !important; /* Hide bottom panel */
}

.show-api {
    display: none !important; /* Hide bottom panel */
}

button, input, textarea, select {
    font-family: Helvetica, Arial, sans-serif; /* Ensure all inputs use Helvetica */
}

.gr-button {
    background-color: #357edd; /* Matte blue button background */
    color: white; /* White text for contrast */
    font-weight: bold; /* Bold font for buttons */
    border: none; /* Remove borders */
    border-radius: 4px; /* Slightly rounded corners for modern look */
}

.gr-button:hover {
    background-color: #78a498; /* Darker matte blue on hover */
}

.gr-textbox, .gr-slider, .gr-checkbox, .gr-dropdown {
    border: 1px solid #357edd; /* Matte blue borders */
    border-radius: 4px; /* Rounded corners */
}

.gr-title, .gr-subtitle {
    font-weight: bold; /* Bold titles */
    color: #357edd; /* Matte blue titles */
}

.container-spec {
    display: flex;
    flex-direction: column;
    width: 100%;
    margin: auto;
}
.assessment {
    padding: 20px;
    font-size: 1.4em; /* Increase font size */
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: bold; /* Bold text */
}
.relevant {
    background-color: #357edd;
    color: #fff !important;
}
.needs-improvement {
    background-color: #ff7f0e;
    color: white !important;
}
.not-relevant {
    background-color: #EE4266;
}
.check-mark {
    font-size: 2em;
}
.prose h1 {
    color: #357edd;
    font-size: 3em !important;
    margin-top: -20px;
}
.prose p {
    font-size: 1.2em !important;
}
.reasoning {
    border: 1px solid #ccc;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    font-size: 1.2em; /* Increase font size */
}
.chat-button {
    height: 100% !important;
}
/* Ensure the text field and button have the same height */
.gr-textbox, .chat-button {
    height: 50px; /* Set the height to match */
    line-height: 50px; /* Align text vertically */
}

/* Ensure the button background matches the theme */
.gr-button, .chat-button {
    background-color: #357edd; /* Matte blue button background */
    color: white; /* White text for contrast */
    font-weight: bold; /* Bold font for buttons */
    border: none; /* Remove borders */
    border-radius: 4px; /* Slightly rounded corners for modern look */
}

/* Hover effect for buttons */
.gr-button:hover, .chat-button:hover {
    background-color: #1E5CAE; /* Darker matte blue on hover */
}
.chat-button {
    height: auto !important;
    padding: 0;
}

.gr-textbox, .gr-button {
    height: 50px; /* Adjust the height value as needed */
    line-height: 50px; /* Match this with the height to vertically center the text */
}

input[type="text"] {
    padding: 0 10px; /* Ensure text is not cut off */
    height: 50px; /* Match the button's height */
    line-height: 50px;
    margin-top: -5px !important;
}"""

demo.launch(share=True, allowed_paths=["workspace/"])
