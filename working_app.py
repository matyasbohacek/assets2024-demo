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
import os

# EXAMPLES should now only hold reference images and videos
examples = [
    ("./examples/bad.png", "./examples/bad.mp4"),
    ("./examples/great.png", "./examples/great.mp4"),
    ("./examples/mid.png", "./examples/mid.mp4"),
]

EXAMPLES = {
    "hello": ("./examples/mid.png", ""),
    "think": ("./examples/THINK.png", ""),
    "phone": ("./examples/PHONE.png", "")
}

def generate_html_analysis(overall_assessment, reasoning_handshape, reasoning_movement, sign):
    # Determine the class and symbol based on the overall assessment
    if "relevant" in overall_assessment:
        assessment_class = "KEEP IT UP!"
        check_handshape = True
        check_movement = True
        symbol = "✓"
        color = "#78a498"
    elif "need" or "Need" in overall_assessment and "improvement" in overall_assessment:
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

    print(f"Image 1 path: assets/PoseGPT/tmp/highest_prob.png")
    path_sign = f"tmp/{sign}"
    print(path_sign)


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
            <img src="/file=tmp/highest_prob.png" alt="Image 1" style="width: 48%; border-radius: 10px;">
            <img src="/file={path_sign}" alt="Image 2" style="width: 48%; border-radius: 10px;">
        </div>
    </div>
    """

    return html_output


def extract_middle_frame(video_path, output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the middle frame index
    middle_frame_index = total_frames // 2
    
    # Set the position of the video to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    # Read the middle frame
    ret, frame = cap.read()
    
    # If the frame was successfully read, save it as an image
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Middle frame saved as {output_image_path}")
    else:
        print("Failed to extract frame")
    
    # Release the video capture object
    cap.release()

    return output_image_path



def merge_images(image1, image2, output_path):
    # Resize both images to 512x512
    target_size = (512, 512)
    image1_resized = image1.resize(target_size)
    image2_resized = image2.resize(target_size)

    # Convert PIL Images to OpenCV format
    image1_cv = cv2.cvtColor(np.array(image1_resized), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.array(image2_resized), cv2.COLOR_RGB2BGR)

    # Get dimensions (they will both be 512x512)
    height, width, _ = image1_cv.shape

    # Create a blank image with double the width
    total_width = width * 2
    merged_image = np.zeros((height, total_width, 3), dtype=np.uint8)

    # Place the first image on the left
    merged_image[0:height, 0:width] = image1_cv

    # Place the second image on the right
    merged_image[0:height, width:total_width] = image2_cv

    # Save the merged image using OpenCV
    cv2.imwrite(output_path, merged_image)

    print(f"Merged image saved at {output_path}")

def handle_video(video, sign_choice, chat_history):
    reference_image_url, reference_video_path = EXAMPLES[sign_choice]
    sign = os.path.basename(reference_image_url)
    args = parse_args()
    
    
    save_path = "./tmp/highest_prob.png"
    
    
    user_key_frame_path = extract_middle_frame(video, save_path)
    # Load the reference image
    reference_image = load_image(reference_image_url)
    user_key_frame = load_image(user_key_frame_path)
    merged_image_path = "./tmp/merged_image.png"
    # Merge the images side by side
    merged_image = merge_images(user_key_frame, reference_image, merged_image_path)

    # Call the chat_with_model function
    overall_assessment, _ = chat_with_model(merged_image_path, "Using only one of these words — relevant, needs-improvement, or not-good — indicate how similar the handshapes are in the two images. Do not include any other text.", args)
    print(f"Overall assessment: {overall_assessment}")
    hand_shape, _ = chat_with_model(merged_image_path, "In one clear sentence, describe exactly how the person on the left should adjust their hand(s) to replicate the handshape of the person on the right.", args)
    print(f"Hand shape: {hand_shape}")
    hand_movement, _ = chat_with_model(merged_image_path, "In one clear sentence, describe precisely how the person on the left should position and move her hand(s) to replicate the hand movement of the person on the right.", args)
    print(f"Hand movement: {hand_movement}")
    # Generate the HTML content
    html_content = generate_html_analysis(
        overall_assessment,  # LLM generated assessment 
        hand_shape,  
        hand_movement,  
        sign
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
            sign_choice = gr.Dropdown(label="Choose a Sign to Perform", choices=["hello", "think", "phone"])
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

demo.launch(share=True, allowed_paths=["."])
