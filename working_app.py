import gradio as gr
from PIL import Image
import sys
import cv2
import numpy as np
import random
from io import BytesIO
import torch
from working_utils import chat_with_model, load_image, parse_args # Import specific utility functions
#from working_utils import chat_with_model, load_image, parse_args, keyframe_extraction # only if using sign-segmentation
import os

# Specific examples for different signs
EXAMPLES = {
    "hello": ("./examples/HELLO.png", ""),
    "think": ("./examples/THINK.png", ""),
    "phone": ("./examples/PHONE.png", "")
}

def generate_html_analysis(overall_assessment, reasoning_handshape, reasoning_movement, sign):
    # Set assessment visual elements based on assessment outcome
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
    else:  # If not relevant
        assessment_class = "not-relevant"
        check_handshape = False
        check_movement = False
        symbol = "✗"
        color = "#EE4266"

    path_sign = f"tmp/{sign}"  # Define path to sign image

    # HTML structure for visual feedback display
    html_output = f"""
    <div class="container-spec">
        <div id="overallAssessment" class="assessment {overall_assessment}" style="background-color: {color};">
            <div style='color: #fff !important; font-weight: bold;'>{assessment_class}</div>
            <div class="check-mark" style='color: #fff !important;'>{symbol}</div>
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
    # Capture the video and extract middle frame
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    # Read and save the middle frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Middle frame saved as {output_image_path}")
    else:
        print("Failed to extract frame")
    
    cap.release()
    return output_image_path

def merge_images(image1, image2, output_path):
    # Resize images and merge them side by side for comparison
    target_size = (512, 512)
    image1_resized = image1.resize(target_size)
    image2_resized = image2.resize(target_size)

    # Convert to OpenCV format
    image1_cv = cv2.cvtColor(np.array(image1_resized), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.array(image2_resized), cv2.COLOR_RGB2BGR)

    # Create blank image and merge
    height, width, _ = image1_cv.shape
    merged_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
    merged_image[0:height, 0:width] = image1_cv
    merged_image[0:height, width:] = image2_cv

    # Save the merged image
    cv2.imwrite(output_path, merged_image)
    print(f"Merged image saved at {output_path}")

def handle_video(video, sign_choice, chat_history):
    # Process video and reference image based on user input
    reference_image_url, reference_video_path = EXAMPLES[sign_choice]
    sign = os.path.basename(reference_image_url)
    args = parse_args()

    #models for sign-segmentation
    # i3d = "./sign-segmentation/models/i3d/i3d_kinetics_bslcp.pth.tar"
    # ms-tcn = "./sign-segmentation/models/ms-tcn/mstcn_bslcp_i3d_bslcp.model"
    
    # Extract middle frame from user video
    user_key_frame_path = extract_middle_frame(video, "./tmp/highest_prob.png")
    #user_key_frame_path = keyframe_extraction(video, i3d, ms-tcn, "./tmp/highest_prob.png") # only if using sign-segmentation 
    reference_image = load_image(reference_image_url)
    user_key_frame = load_image(user_key_frame_path)
    
    # Merge images and assess handshape and movement
    merge_images(user_key_frame, reference_image, "./tmp/merged_image.png")
    overall_assessment, _ = chat_with_model("./tmp/merged_image.png", "Using only one of these words — relevant, needs-improvement, or not-good — indicate how similar the handshapes are in the two images.", args)
    hand_shape, _ = chat_with_model("./tmp/merged_image.png", "In one clear sentence, describe exactly how the person on the left should adjust their hand(s) to replicate the handshape of the person on the right.", args)
    hand_movement, _ = chat_with_model("./tmp/merged_image.png", "In one clear sentence, describe precisely how the person on the left should position and move her hand(s) to replicate the hand movement of the person on the right.", args)

    # Generate HTML content to display assessment
    html_content = generate_html_analysis(overall_assessment, hand_shape, hand_movement, sign)
    chat_history.clear()  # Reset chat history

    return html_content, chat_history

def handle_chat_input(chat_input, chat_history):
    # Handle user text input for further assessment feedback
    args = parse_args()
    response, _ = chat_with_model("/tmp/merged_image.png", chat_input, args)
    chat_history.append((chat_input, response))
    return chat_history

# Gradio interface for the app
with gr.Blocks(css=".btn {background-color: #357edd;}", theme=gr.themes.Soft(font=["-apple-system", "BlinkMacSystemFont", "sans-serif"])) as demo:
    gr.Markdown("# Sign Language Feedback")
    gr.Markdown("Choose a sign you want to perfect and sign it on the camera!")

    with gr.Row():
        with gr.Column():
            sign_choice = gr.Dropdown(label="Choose a Sign to Perform", choices=["hello", "think", "phone"])
            video_input = gr.Video(label="Upload Sign Language Video")
            submit_button = gr.Button("Submit Video", elem_id="submit-button")
        with gr.Column():
            output_html = gr.HTML(visible=True)  # Visible HTML output from the start
            with gr.Accordion("Reference", open=False, visible=False) as reference_vid_wr:
                reference_video = gr.Video(label="Reference Video", visible=False)  # Hidden reference video
            with gr.Accordion("Follow-up Chat", open=False, visible=False) as chat_module:
                chatbot = gr.Chatbot(label="Chat Interface", height=300)  # Chat for feedback
                with gr.Row():
                    with gr.Column(scale=4):
                        chat_input = gr.Textbox(placeholder="Type your message here...", show_label=False)
                    with gr.Column(scale=1):
                        chat_submit = gr.Button("Send", elem_classes="chat-button")

    submit_button.click(handle_video, inputs=[video_input, sign_choice, chatbot], outputs=[output_html, chatbot])
    chat_submit.click(handle_chat_input, inputs=[chat_input, chatbot], outputs=[chatbot])

demo.launch(share=True, allowed_paths=["."])
