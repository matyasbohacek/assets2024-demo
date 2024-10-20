import argparse
import os
import sys

from PIL import Image
import requests
import math
from pathlib import Path
import sign_utils
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig

from model.chatpose import ChatPoseForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import json
from tqdm import tqdm
from io import BytesIO
from transformers import TextStreamer

import torch.nn as nn

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="ChatPose chat")
    
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--version", default="YaoFeng/CHATPOSE-V0")
    parser.add_argument("--image_file", type=str, default=None)
    parser.add_argument("--image_dir", default="./dataset/Yoga-82")
    parser.add_argument("--json_path", default="./dataset/Yoga-82/yoga_dataset.json")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--out_dim", default=144, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument(
        "--dataset", default="hmr||vqa", type=str
    )
    parser.add_argument("--text_embeddings_for_global", action="store_true", default=False)
    parser.add_argument("--predict_global_orient", action="store_true", default=False)
    parser.add_argument("--cat_image_embeds", action="store_true", default=False)
    
    # If args is None, pass an empty list to use default values
    if args is None:
        args = []
    
    return parser.parse_args(args)

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
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

def chat_with_model(image_file, text_input, args):
    args.image_file = image_file
    if args.exp_name is not None:
        save_name = args.exp_name.upper()
        args.version = f"./checkpoints/{save_name}"
        args.vis_save_path = f"./vis_output/{save_name}"
    else:
        save_name = args.version.split('/')[-1]
        args.vis_save_path = f"./vis_output/{save_name}"
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Add special tokens to the tokenizer
    special_tokens_dict = {
        'additional_special_tokens': [
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            '[SEG]',
            '[SEG_END]'
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer_vocab_size = len(tokenizer)
    args.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update({"out_dim": args.out_dim})
    kwargs.update({"text_embeddings_for_global": args.text_embeddings_for_global})
    kwargs.update({"predict_global_orient": args.predict_global_orient})
    kwargs.update({"cat_image_embeds": args.cat_image_embeds})
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Load model after tokenizer is updated
    model = ChatPoseForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs
    )

    # Resize model embeddings after adding special tokens
    model.resize_token_embeddings(tokenizer_vocab_size)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = vision_tower.image_processor
    model.eval()

    if not os.path.exists(image_file):
        return "Image File not found, use ChatPose without image input", None

    # Load and preprocess the image
    image_np = cv2.imread(image_file)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # Preprocess for the vision tower
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    # Prepare the image for other processing (e.g., resizing)
    image = image_clip.clone()
    image = F.interpolate(
        image.float(), size=[256, 256], mode='bilinear', align_corners=False
    ).to(image_clip.dtype)

    # Initialize the conversation
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    roles = conv.roles

    questions = [text_input]
    answers = []
    visualizations = []

    # Prepare the prompt using the correct special tokens
    prompt = text_input
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    # Generate the response
    output_ids, predictions, pred_smpl_params = model.evaluate(
        image_clip,
        image,
        input_ids,
        max_new_tokens=512,
        tokenizer=tokenizer,
        return_smpl=True,
    )

    # Decode the output
    output_ids = output_ids[0, input_ids.shape[1]:]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip().replace("</s>", "")
    text_output = text_output.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]")
    conv.messages[-1][-1] = text_output
    answers.append(text_output)
    visualizations.append(predictions)

    # Save the visualization
    imagename = os.path.basename(image_file)
    save_path = os.path.join(args.vis_save_path, imagename)
    visualize_LLM(save_path, questions, answers, visualizations)

    return text_output, save_path

#Sign segmentation

def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        video_path = Path(video_path)
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
               f"-filter:v fps=fps={fps} {video_path}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    f = 0
    rgb = []
    while True:
        # frame: BGR, (h, w, 3), dtype=uint8 0..255
        ret, frame = cap.read()
        if not ret:
            break
        # BGR (OpenCV) to RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
        f += 1
    cap.release()
    # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)

    return rgb

def load(video_path: str, fps: int = 25) -> (torch.Tensor, list):
    cap = cv2.VideoCapture(video_path)
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)

    cap.release()
    return frame_list



def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3), std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        tmp = resize_generic(
            im_to_numpy(tmp), resize_res, resize_res, interp="bilinear", is_flow=False
        )
        rgb_resized[t] = tmp
    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb


def load_i3d_model(
        i3d_checkpoint_path: Path,
        num_classes: int,
        num_in_frames: int,
) -> torch.nn.Module:
    """Load pre-trained I3D checkpoint, put in eval mode."""
    model = sign_utils.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
        include_embds=True,
    )
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(i3d_checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def load_mstcn_model(
        mstcn_checkpoint_path: Path,
        device,
        num_blocks: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        dim: int = 1024,
        num_classes: int = 2,

) -> torch.nn.Module:
    """Load pre-trained MS-TCN checkpoint, put in eval mode."""
    model = sign_utils.MultiStageModel(
        num_blocks, 
        num_layers, 
        num_f_maps, 
        dim, 
        num_classes,
    )

    model = model.to(device)
    checkpoint = torch.load(mstcn_checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def sliding_windows(
        rgb: torch.Tensor,
        num_in_frames: int = 1,
        stride: int = 1,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided


def main_i3d(
    i3d_checkpoint_path: Path,
    video_path: Path,
    fps: int = 25,
    num_classes: int = 981,
    num_in_frames: int = 1,
    batch_size: int = 1,
    stride: int = 1,
):
    model = load_i3d_model(
        i3d_checkpoint_path=i3d_checkpoint_path,
        num_classes=num_classes,
        num_in_frames=num_in_frames,
    )
    rgb_orig = load_rgb_video(
        video_path=video_path,
        fps=fps,
    )
    # Prepare: resize/crop/normalize
    rgb_input = prepare_input(rgb_orig)
    # Sliding window
    rgb_slides = sliding_windows(
        rgb=rgb_input,
        stride=stride,
        num_in_frames=num_in_frames,
    )
    # Number of windows/clips
    num_clips = rgb_slides.shape[0]
    # Group the clips into batches
    num_batches = math.ceil(num_clips / batch_size)
    all_features = torch.Tensor(num_clips, 1024)
    all_logits = torch.Tensor(num_clips, num_classes)
    for b in range(num_batches):
        inp = rgb_slides[b * batch_size : (b + 1) * batch_size]
        # Forward pass
        out = model(inp)
        logits = out["logits"].data.cuda()
        all_features[b] = out["embds"].squeeze().data.cuda()
        all_logits[b] = logits.squeeze().data.cuda()
    
    return all_features, all_logits

def main_mstcn(
    features,
    logits,
    mstcn_checkpoint_path: Path,
):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = load_mstcn_model(
        mstcn_checkpoint_path=mstcn_checkpoint_path,
        device=device
    )


    sm = nn.Softmax(dim=1)

    # Number of windows/clips
    num_clips = features.shape[0]//100
    # Group the clips into batches
    num_batches = math.ceil(num_clips)

    all_preds = []
    all_probs = []

    for b in range(num_batches+1):
        inp = features[b * 100 : (b + 1) * 100]
        inp = np.swapaxes(inp, 0, 1)
        inp = inp.unsqueeze(0).to(device)
        predictions = model(inp, torch.ones(inp.size(), device=device))
        pred_prob = list(sm(predictions[-1]).cpu().detach().numpy())[0][1]
        predicted = torch.tensor(np.where(np.asarray(pred_prob) > 0.5, 1, 0))

        all_preds.extend(torch_to_list(predicted))
        all_probs.extend(pred_prob)

    return all_probs


def keyframe_extraction(video_path, i3d_checkpoint_path, mstcn_checkpoint_path, save_path):
    features, logits = main_i3d(i3d_checkpoint_path=i3d_checkpoint_path, video_path=video_path)
    all_probs = main_mstcn(features, logits, mstcn_checkpoint_path)
    all_probs = [float(val) for val in all_probs]
    highest_prob = max(all_probs)
    print(highest_prob)
    highest_prob_ind = all_probs.index(highest_prob)
    print(highest_prob_ind)

    frames = load(video_path)

    highest_prob_frame = frames[highest_prob_ind]
    save_path = save_path
    cv2.imwrite(save_path, highest_prob_frame)

    return save_path
