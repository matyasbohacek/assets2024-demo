import os
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer
from model.chatpose import ChatPoseForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def load_image(image_file):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def visualize_LLM(save_path, questions, answers, visualizations):
    import matplotlib.pyplot as plt
    questions = [f'\nQ{i}: {question}' for i, question in enumerate(questions)]
    answers = [f'\nA{i}: {answer.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]")}' for i, answer in enumerate(answers)]

    if len(questions) == 1:
        questions.append('')
        answers.append('')
        visualizations.append(None)
    
    fig, ax = plt.subplots(nrows=len(questions), ncols=1, figsize=(12, 12))

    for k, ax_ in enumerate(ax.flat):
        if visualizations[k] is not None:
            image = (np.transpose(visualizations[k].cpu().numpy(), (1,2,0))*255).astype(np.uint8)
            ax_.imshow(image)
        ax_.axis('off')
        ax_.set_title(questions[k] + '\n' + answers[k], wrap=True, color='g')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def handle_video(video_input, chatbot):
    args = parse_args([])
    video_path = video_input.name

    image_path = args.image_file if args.image_file else video_path
    if not os.path.exists(image_path):
        return [(None, "Image file not found, use ChatPose without image input")]

    image_np = load_image(image_path)
    image_clip = preprocess(torch.from_numpy(image_np).permute(2, 0, 1).float())
    image_clip = image_clip.unsqueeze(0).cuda()

    args.image_file = image_path
    save_name = args.exp_name.upper() if args.exp_name else args.version.split('/')[-1]
    args.vis_save_path = f"./vis_output/{save_name}"
    os.makedirs(args.vis_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    model = ChatPoseForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, torch_dtype=torch_dtype, out_dim=args.out_dim)

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype).cuda()
    model.eval()

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    roles = conv.roles
    questions = []
    answers = []
    visualizations = []

    def update_chat(inp, chatbot):
        questions.append(inp)
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + inp if args.use_mm_start_end else inp
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        output_ids, predictions, pred_smpl_params = model.evaluate(
            image_clip,
            image_clip,
            input_ids,
            max_new_tokens=512,
            tokenizer=tokenizer,
            return_smpl=True,
        )
        output_ids = output_ids[0, input_ids.shape[1]:]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip().replace("</s>", "")
        text_output = text_output.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]")
        conv.messages[-1][-1] = text_output
        answers.append(text_output)
        visualizations.append(predictions)
        chatbot.append(("Bot", text_output))

        imagename = os.path.basename(image_path)
        save_path = os.path.join(args.vis_save_path, imagename)
        visualize_LLM(save_path, questions, answers, visualizations)
        
        return chatbot

    return update_chat

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

    submit_button.click(handle_video, inputs=[video_input, chatbot], outputs=[chatbot])

    demo.load(preload_answers, inputs=None, outputs=[chatbot])

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

demo.launch()

