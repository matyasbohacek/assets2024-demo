import gradio as gr

EXAMPLES = [
    (
        "LET'S IMPROVE THIS",
        True,
        "Your handshape looks good, keep it up!",
        False,
        "Make sure you touch your forehead. See reference frame below.",
        "https://data.matsworld.io/mid.png",
        "https://data.matsworld.io/reference.png",
        "examples/reference.mp4"
    ),
    (
        "KEEP IT UP",
        True,
        "Your handshape looks good, keep it up!",
        True,
        "The hand movement execution looks great!",
        "https://data.matsworld.io/great.png",
        "https://data.matsworld.io/reference.png",
        "examples/reference.mp4"
    ),
    (
        "LET'S START OVER",
        False,
        "You should change the hand shape to be an open palm.",
        False,
        "Make sure you touch your forehead. See reference frame below.",
        "https://data.matsworld.io/bad.png",
        "https://data.matsworld.io/reference.png",
        "examples/reference.mp4"
    ),
]

index_ex = 2


# Define a function to handle video input and generate responses
def handle_video(video, sign_choice, chat_history):
    overall_assessment = EXAMPLES[index_ex][0]  # Placeholder assessment
    check_handshape = EXAMPLES[index_ex][1]  # Placeholder check
    reasoning_handshape = EXAMPLES[index_ex][2]  # Placeholder reasoning
    check_movement = EXAMPLES[index_ex][3]  # Placeholder check
    reasoning_movement = EXAMPLES[index_ex][4]  # Placeholder reasoning
    image_path_1 = EXAMPLES[index_ex][5]  # Placeholder image path
    image_path_2 = EXAMPLES[index_ex][6]  # Placeholder image path
    reference_video_path = EXAMPLES[index_ex][7]  # Placeholder reference video path

    # Generate HTML analysis
    _, html_content = generate_html_analysis(
        overall_assessment,
        check_handshape,
        reasoning_handshape,
        check_movement,
        reasoning_movement,
        image_path_1,
        image_path_2
    )

    # The chat history remains empty as requested
    chat_history.clear()

    placeholder_html = f"<p>Processing video for '{sign_choice}'. Please wait...</p>"
    return html_content, chat_history, gr.update(visible=True), gr.update(visible=True, value=reference_video_path), gr.update(visible=True)


def handle_chat_input(chat_input, chat_history):
    chat_history.append((chat_input, "Yes, open your palm more"))
    return chat_history, ""


with gr.Blocks(css=".btn {background-color: #357edd;}",
               theme=gr.themes.Soft(font=["-apple-system", "BlinkMacSystemFont", "sans-serif"])) as demo:
    gr.Markdown("<br>")
    gr.Markdown("# Sign Language Feedback")
    gr.Markdown("Choose a sign you want to perfect and sign it on the camera!")

    with gr.Row():
        with gr.Column():
            sign_choice = gr.Dropdown(label="Choose a Sign to Perform", choices=["Apple", "Coffee", "Hello"])
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

    submit_button.click(handle_video, inputs=[video_input, sign_choice, chatbot],
                        outputs=[output_html, chatbot, chat_module, reference_video, reference_vid_wr])

    chat_submit.click(handle_chat_input, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])


def generate_html_analysis(overall_assessment, check_handshape, reasoning_handshape, check_movement, reasoning_movement,
                           image_path_1, image_path_2):
    # Determine the class and symbol based on the overall assessment
    if overall_assessment == "KEEP IT UP":
        assessment_class = "relevant"
        symbol = "✓"
        color = "#78a498"
    elif overall_assessment == "LET'S IMPROVE THIS":
        assessment_class = "needs-improvement"
        symbol = "!"
        color = "#ff7f0e"
    else:  # "MORE WORK TO DO"
        assessment_class = "not-relevant"
        symbol = "✗"
        color = "#EE4266"

    # Create HTML structure
    html_output = f"""
    <div class="container-spec">
        <div id="overallAssessment" class="assessment {assessment_class}" style="background-color: {color};">
            <div style='color: #fff !important; font-weight: bold;'>{overall_assessment}</div><div class="check-mark" style='color: #fff !important;'>{symbol}</div>
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

    summary_text = f"""
    Overall Assessment: {overall_assessment}
    Handshape: {"LOOKS GOOD" if check_handshape else "MIGHT BE PROBLEMATIC"} {reasoning_handshape}
    Hand Movement: {"LOOKS GOOD" if check_movement else "MIGHT BE PROBLEMATIC"} {reasoning_movement}
    """

    return summary_text, html_output

# Apply CSS for styling
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

demo.launch()
