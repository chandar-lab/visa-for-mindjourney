import base64
import copy
from PIL import Image
from utils.InternVL3 import *
from typing import List, Optional, Dict

SYS = """
You are an AI assistant designed to help us understand spatial relationship in 3D indoor scene and finish visual question answering.
"""

BASELINE_PROMPT = """
You will be given one or two images and a spatial reasoning question.
Your goal is to answer the spatial related question correctly.

Directly output an answer from the answer choices provided below.
You can add some analysis in your response, but remember to format the end of your answer according to the rule.

Now, according to the following image, answer the question from provided choices:
Question: {question}
Answer Choice: {answer_choice}

Answer: 
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_gpt_content(contents):
    formatted_content = []
    for c in contents:
        formatted_content.append({"type": "text", "text": c[0]})
        if len(c) == 2: # has image
            formatted_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(c[1])}",
                        "detail": "high",
                    },
                }
            )
    return formatted_content

def format_internvl3_content(contents, model_device=None):
    formatted_content = {
        "question": "",
        "num_patches_list": [],
    }
    pixel_values = []
    for c in contents:
        formatted_content["question"] += c[0]
        formatted_content["question"] += "\n"
        if len(c) == 2: # has image
            formatted_content["question"] += "<image>\n"
            img_tensor = load_image(c[1], max_num=12).to(torch.bfloat16)
            if model_device is not None:
                img_tensor = img_tensor.to(model_device)
            else:
                img_tensor = img_tensor.cuda()

            pixel_values.append(img_tensor)
            formatted_content["num_patches_list"].append(img_tensor.size(0))
    formatted_content["pixel_values"] = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
    return formatted_content

def format_qwen3vl_content(contents, model_device=None):
    """
    Format content for Qwen3-VL model.
    Returns messages in the format expected by Qwen3-VL.
    """
    messages = []
    current_message = {"role": "user", "content": []}
    
    for c in contents:
        if len(c) == 2:  # has image
            # Add image and text to the current message
            current_message["content"].extend([
                {"type": "image", "image": f"file://{c[1]}"},
                {"type": "text", "text": c[0]}
            ])
        else:  # text only
            current_message["content"].append({"type": "text", "text": c[0]})
    
    if current_message["content"]:
        messages.append(current_message)
    
    return messages

def format_spatial_vqa_prompt_answer_baseline(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
            The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"{choice}\n"
    content.append((q_text,))
    content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_scaling(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict,
    action_claims: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }
        action_claims (dict, optional): A nested dictionary of validated claims for each action consequence.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": ["validated_claim1", "validated_claim2", ...],
                        "subaction_2": ["validated_claim1", "validated_claim2", ...],
                        ...
                    },
                    "action_2": {
                        "subaction_1": ["validated_claim1", "validated_claim2", ...],
                        "subaction_2": ["validated_claim1", "validated_claim2", ...],
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    if action_claims:
        sys_prompt = (
            "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
            "You must analyze any provided images or observations and answer the question.\n\n"
            "Rules:\n"
            "1. You should output the exact answer from the choices.\n"
            "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
            "3. When validated claims are provided for imagined views, use these claims as reliable observations to help answer the question.\n"
            "4. Your final line must only include the exact answer choice.\n"
        )
    else:
        sys_prompt = (
            "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
            "You must analyze any provided images or observations and answer the question.\n\n"
            "Rules:\n"
            "1. You should output the exact answer from the choices.\n"
            "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
            "3. Your final line must only include the exact answer choice.\n"
        )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    if action_claims:
        actions_intro = (
            "Below are the imagined views you would obtain if you took the corresponding actions. "
            "For each imagined view, validated claims about what changed with respect to the egocentric view are also provided. "
            "These are provided to help you answer the question.\n"
            "Use the validated claims as reliable observations in your reasoning, but you should still only output the exact answer at the last line\n"
        )
    else:
        actions_intro = (
            "Below are the imagined views you would obtain if you took the corresponding actions. "
            "These are provided to help you answer the question.\n"
            "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
        )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
            
            # Add validated claims if available
            if action_claims and action_str in action_claims and subaction_str in action_claims[action_str]:
                claims = action_claims[action_str][subaction_str]
                if claims:
                    content.append(("Validated observations from this view:\n",))
                    for claim in claims:
                        content.append((f"  - {claim}\n",))
                    content.append(("\n",))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_scores(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the imaginations during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    content.append((q_text,))

    if len(answer_choices) > 0:
        ac_text = "Answer Choices:\n"
        for choice in answer_choices:
            ac_text += f"  - {choice}\n"
        ac_text += "\n"
        content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_baseline_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list = None,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. Your final line must only include the exact answer choice.\n"
        "3. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        r"4. The final answer MUST BE put in \boxed{}."
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"{choice}\n"
    content.append((q_text,))
    # content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer in a float-point number format.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_scores_fill_in_blank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_rank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
) -> (str, list):
    
    """
    Rank the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images and rank indexes of imagined images from most relevant to least relevant.\n\n"
        "Rules:\n"
        "1. You'll be provided with images (including imagined images), a question, and a set of answer choices. You should rank most relevant images that can help you answer the question from the choices.\n"
        "2. You should output a list of indexes, separated by ','. For example: Output: 3,1,2,0\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            encoded_img = encode_image(img_path)
            content.append((f"Image {idx + 1}:", encoded_img))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        encoded_img = encode_image(img_path)
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", encoded_img))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of indexes from most relevant image to least relevant image.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_scaling_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. You can include minimal reasoning, but your final line must only include the exact answer.\n"
        "4. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        "5. The final answer MUST BE put in \boxed{}."
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions.\n"
        "If there are more than one image in the question, these imaged views are based on the first image.\n"
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output the exact answer from the question.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_bbox(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze the image and answer the question.\n\n"
        "Rules:\n"
        "1. Output the bounding box in your current egocentric view of the area most important and relevant for answering the question. For those questions containing marks, it is important to have the bounding box include the object that marked with the number mentioned in the question.\n"
        "2. The output should only contain two integer coordinates of the top-left and bottom-right corners of the bounding box, separated by ':' in the format (x1,y1):(x2,y2).\n"
        "3. Only output None if you are very uncertain about the bounding box location or it is not necessary for answering the question. This case is rare to happen.\n"
    )
    content = []
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view of size {Image.open(images[0]).size}\n",))
    else:
        content.append(("No image provided.\n\n",))
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    content.append((q_text,))
    content.append((ac_text,))
    instructions = (
        "Output either the bounding box coordinates in the format (x1,y1):(x2,y2) or None if uncertain or not needed.\n"
        "Output: "
    )
    content.append((instructions,))
    return sys_prompt, content


def format_verification_prompt_claim_generation(
    action_description: str,
    frames: list,
    frame_range: str,
    frame_indices: List[int],
    question: str,
) -> (str, list):
    """
    Format prompt for generating micro-claims from video frames.
    """
    sys_prompt = (
        "You are an AI assistant that generates micro-claims about spatial observations in video frames. "
        "Your task is to describe what you observe in the provided frames after a camera action. "
        "Focus ONLY on the spatial relationships and changes that are relevant to answering the spatial reasoning question."
        "Generate EXACTLY 2-3 claims, no more, no less."
    )
    
    content = [
        (f"After performing the action: '{action_description}'",),
        (f"Analyze the following frames ({frame_range}) and generate 2-3 specific micro-claims about what you observe:",),
    ]
    
    # Add frames to content
    for i, frame_path in enumerate(frames):
        this_frame_index = frame_indices[i]
        content.append((f"Frame {this_frame_index}:", frame_path))
    
    content.extend([
        (f"\nThe original question is: {question}",),
        ("\nGenerate micro-claims in this format:",),
        ("- [Specific observation] in {frame_range}",),
        ("- [Spatial relationship] in {frame_range}",),
        ("- [Object property or change] in {frame_range}",),
        ("\nFocus on observations that would help answer the spatial reasoning question.",),
        ("Output only the 2-3 micro-claims, one per line, starting with '- '",)
    ])
    
    return sys_prompt, content

def format_image_comparison_prompt(
                                   action_description: str,
                                   images: List[str],
                                   question: str,
                                   n_claims: int = 3,
                                   answer_choices: Optional[List[str]] = None) -> (str, List):
    """
    Format prompt for generating claims from image comparison.
    """
    # sys_prompt = (
    #     "You are an AI assistant that generates atomic frame-anchored micro-claims about spatial observations when comparing two images. "
    #     "Your task is to identify clear, objective changes between the 'before' and 'after' images that result from a camera action. "
    #     "Focus ONLY on spatial relationships and changes that are relevant to answering the spatial reasoning question. "
    #     "Generate EXACTLY 2-4 claims that are necessary, reliable, and easy to verify."
    #     "It's better to generate fewer high-quality claims than many uncertain ones."
    # )
    sys_prompt = (
        "You are an AI assistant that generates atomic, frame-anchored micro-claims about spatial observations when comparing two images. "
        "Your primary goal is to identify changes that help distinguish between specific answer choices for a spatial reasoning question. "
        "Generate claims that are directly relevant to the question, objectively verifiable, and useful for decision-making. "
        "Focus on binary, measurable changes rather than subjective observations. "
        "Generate EXACTLY 2-4 high-quality claims that would help a human choose between the answer options."
    )

    # Extract action type for targeted guidance
    action_type = "movement"
    if "turn" in action_description.lower():
        action_type = "rotation"
    elif "forward" in action_description.lower():
        action_type = "forward movement"
    
    content = [
        (f"After performing the action: '{action_description}'",),
        (f"Compare these two images and generate 2-3 specific micro-claims about what changed:",),
        (f"BEFORE (previous view):", images[0]),
        (f"AFTER (current view):", images[1]),
        (f"\nThe original question is: {question}",),
    ]
    
    # Add answer choices if provided
    if answer_choices:
        print(f"answer choices provided for claim generation prompt: {answer_choices}")
        ac_text = "Answer Choices:\n"
        for choice in answer_choices:
            ac_text += f"  - {choice}\n"
        ac_text += "\nFocus your claims on observations that would help distinguish between these answer choices.\n"
        content.append((ac_text))
    
    content.extend([
        (f"\nFor this {action_type} action, prioritize these types of changes:",),
        ("1. VISIBILITY CHANGES: Objects appearing/disappearing, moving in/out of frame",),
        ("2. EDGE POSITIONING: Objects moving closer to or away from frame edges",),
        ("3. RELATIVE POSITIONS: Clear positional shifts with reference to other objects or frame boundaries",),
        ("4. OCCLUSION CHANGES: Objects becoming more or less hidden by other objects",),
        
        ("\nGuidelines for claim generation:",),
        ("- Generate 2-4 claims that directly help distinguish between the answer choices",),
        ("- If changes are minimal or unclear, generate fewer claims (quality over quantity)",),
        ("- Each claim should be independently verifiable and distinct",),
        ("- Prioritize claims that would help a human choose between the answer options",),
        ("- Focus on binary, measurable changes rather than subjective observations",),
        
        ("\nExamples of HIGH-QUALITY claim patterns:",),
        ("- [Object X] [appears/disappears] on the [left/right] side of the frame",),
        ("- [Object X] moves [closer to/further from] the [left/right] edge",),
        ("- [Object X] becomes [more/less] visible on the [left/right] side",),
        ("- [Object X] moves [toward/away from] [reference object] in the frame",),
        ("- [Object X] [enters/exits] the frame on the [left/right] side",),
        
        ("\nAVOID these LOW-QUALITY claim types:",),
        ("- 'remains the same' or 'stays in position' (not useful for decision-making)",),
        ("- Size changes ('appears larger/smaller') as these are subjective",),
        ("- Vague movements ('shifts leftward') without clear reference points",),
        ("- Quality assessments ('looks closer/farther') without measurable changes",),
        ("- Multi-predicate sentences (split them into separate claims)",),
        ("- 3D depth language (in front/behind) unless occlusion clearly changes",),
        ("- Subjective descriptions ('slightly', 'somewhat', 'a bit')",),
        
        ("\nRequirements:",),
        ("- Be specific about locations and reference points",),
        ("- Focus on binary or clear changes (visible/not visible, in frame/out of frame)",),
        ("- Use precise directional language (left edge, right side, top corner)",),
        ("- Make claims that would help answer the specific spatial reasoning question",),
        ("- Generate only claims you can confidently verify",),
        ("- Ensure each claim is relevant to distinguishing between the answer choices",),
        
        ("\nOutput format:",),
        ("- Output the micro-claims, one per line, starting with '- '",),
        ("- Each claim should be a single, clear sentence",),
        ("- Focus on the most important changes for answering the question",),
    ])
    return sys_prompt, content

def format_verification_prompt_claim_verification(
    claim: dict,
    frames: list,
    reason: bool = True,
) -> (str, list):
    """
    Format prompt for verifying micro-claims against video frames using semantic verification.
    """
    sys_prompt = (
        "You are an AI assistant that verifies micro-claims against visual frames using semantic reasoning. "
        "Your task is to determine the logical relationship between a claim and the visual evidence. "
        "Focus on whether the evidence entails, contradicts, or provides insufficient information about the claim. "
        "Be precise and consider the semantic meaning of the claim in relation to what you observe."
        "Additionally, provide a confidence score reflecting how certain you are about your judgment."
    )
    
    content = [
        (f"Verify this micro-claim against the provided frames:",),
        (f"Claim: '{claim['text']}'",),
        (f"Frame range: {claim['frame_range']}",),
        ("\nAnalyze the frames and determine the semantic relationship between the claim and evidence:",),
    ]
    
    # Add frames to content
    for i, frame_path in enumerate(frames):
        content.append((f"Frame {i+1}:", frame_path))
    
    content.extend([
        ("\nInstructions:",),
        ("1. Examine the specific frames mentioned in the claim carefully",),
        ("2. Determine if the visual evidence ENTAILS the claim (strongly supports it)",),
        ("3. Check if the evidence CONTRADICTS the claim (directly opposes it)",),
        ("4. Assess if the evidence is INSUFFICIENT (lacks information to determine support or contradiction)",),
        ("5. Consider spatial relationships, object properties, movements, and transformations",),
        ("6. For spatial reasoning tasks, focus on directional movements, rotations, and perspective changes",),
        ("7. Evaluate your confidence in the judgment (0.0 = completely uncertain, 1.0 = completely certain)",),
        ("8. Respond with the required format including verdict, confidence, and reasoning",),
        ("\nResponse format:",),
        ("VERDICT: [ENTAILED/CONTRADICTED/INSUFFICIENT]",),
        ("CONFIDENCE: [0.0-1.0]",),
    ])
    
    # Conditionally add reasoning line based on reason parameter
    if reason:
        content.append(("REASONING: [Clear explanation of the semantic relationship and confidence level]",))
    
    content.extend([
        ("\nConfidence Guidelines (0.0-1.0 scale):",),
        ("- 0.95-1.0: Extremely clear, unambiguous evidence",),
        ("- 0.85-0.94: Very clear evidence with minor uncertainties",),
        ("- 0.70-0.84: Clear evidence with some ambiguity",),
        ("- 0.50-0.69: Moderate evidence, noticeable uncertainty",),
        ("- 0.30-0.49: Weak evidence, significant uncertainty",),
        ("- 0.10-0.29: Very unclear evidence, high uncertainty",),
        ("- 0.00-0.09: No clear evidence or contradictory signals",),
        ("\nImportant: Use the full 0.0-1.0 range. Reserve 0.9+ for truly exceptional cases.",),
    ])
    
    return sys_prompt, content

def format_verification_prompt_claim_verification_with_probs(
    claim: dict,
    frames: list,
) -> (str, list):
    """
    Enhanced verification prompt with single-letter verdicts and probabilities.
    """
    sys_prompt = (
        "You are an AI assistant that verifies micro-claims against visual frames using semantic reasoning. "
        "Your task is to determine the logical relationship between a claim and the visual evidence. "
        "Focus on spatial relationships, object properties, movements, and transformations. "
        "Provide probabilities for each possible relationship (entails, contradicts, insufficient) that sum to 1.0. "
        "Be conservative with high probabilities - reserve 0.9+ for truly exceptional cases."
    )

    content = [
        (f"Verify this micro-claim against the provided frames:",),
        (f"Claim: '{claim['text']}'",),
        (f"Frame range: {claim['frame_range']}",),
        ("\nAnalyze the frames and determine the semantic relationship between the claim and evidence:",),
    ]

    # Add frames
    for i, frame_path in enumerate(frames):
        content.append((f"Frame {i+1}:", frame_path))

    content.extend([
        ("\nInstructions:",),
        ("1. Examine all frames carefully for evidence related to the claim",),
        ("2. Focus on spatial relationships, object positions, movements, and transformations",),
        ("3. Determine if evidence ENTAILS (E), CONTRADICTS (C), or is INSUFFICIENT (I) for the claim",),
        ("4. Assign probabilities that sum to 1.0 (±0.01)",),
        ("5. VERDICT should match the highest probability",),
        ("6. Be conservative - use high probabilities only for very clear cases",),
        ("",),
        ("Response format:",),
        ("VERDICT: E",),
        ("p(E): 0.80",),
        ("p(C): 0.10",),
        ("p(I): 0.10",),
        ("REASONING: Brief and concise explanation",),
        ("",),
        ("Verdict meanings:",),
        ("E = ENTAILED (evidence strongly supports the claim)",),
        ("C = CONTRADICTED (evidence clearly opposes the claim)",),
        ("I = INSUFFICIENT (not enough information to decide)",),
        ("",),
        ("Probability calibration examples:",),
        ("Unambiguous case: p(E): 0.90, p(C): 0.05, p(I): 0.05 - claim clearly visible with no ambiguity",),
        ("Strong evidence: p(E): 0.75, p(C): 0.10, p(I): 0.15 - claim well-supported but minor uncertainties",),
        ("Moderate evidence: p(E): 0.55, p(C): 0.15, p(I): 0.30 - claim likely but notable ambiguities",),
        ("Ambiguous: p(E): 0.35, p(C): 0.15, p(I): 0.50 - insufficient information dominates",),
        ("Contradictory: p(E): 0.05, p(C): 0.85, p(I): 0.10 - evidence clearly contradicts claim",),
        ("Maximum uncertainty: p(E): 0.33, p(C): 0.33, p(I): 0.34 - no clear evidence either way",),
        ("",),
        ("Critical guidelines:",),
        ("- Before answering, take a brief moment to reason internally about what each frame shows.",),
        ("- Check that p(E) + p(C) + p(I) = 1.0",),
        ("- VERDICT should match highest probability",),
        ("- If unsure, use I (INSUFFICIENT) rather than guessing",),
        ("- Consider image quality and clarity in your assessment",),
        ("- Use mid-range probabilities (0.4–0.6) when evidence is ambiguous or partially visible.",),
        ("- Probabilities represent your subjective confidence, not mathematical certainty.",),
        ("",),
    ])

    return sys_prompt, content


def format_spatial_vqa_prompt_answer_baseline_open_ended(
    question: str,
    images: list,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for open-ended spatial VQA.
    
    Args:
        question (str): The question to answer.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant's overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question
    q_text = f"Question: {question}\n"
    content.append((q_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_scaling_open_ended(
    question: str,
    images: list,
    action_consequences: dict,
    action_claims: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> (str, list):
    """
    Format a ChatGPT prompt for open-ended spatial VQA with imagined actions.
    
    Arguments:
        question (str): The question to answer.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
        action_claims (dict, optional): A nested dictionary of validated claims for each action consequence.

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    if action_claims:
        sys_prompt = (
            "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
            "You must analyze any provided images or observations and answer the question.\n\n"
            "Rules:\n"
            "1. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
            "2. When validated claims are provided for imagined views, use these claims as reliable observations to help answer the question.\n"
        )
    else:
        sys_prompt = (
            "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
            "You must analyze any provided images or observations and answer the question.\n\n"
            "Rules:\n"
            "1. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question ------------------
    q_text = f"Question: {question}\n\n"
    content.append((q_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    if action_claims:
        actions_intro = (
            "Below are the imagined views you would obtain if you took the corresponding actions. "
            "For each imagined view, validated claims about what changed with respect to the egocentric view are also provided. "
            "These are provided to help you answer the question.\n"
            "Use the validated claims as reliable observations in your reasoning.\n"
        )
    else:
        actions_intro = (
            "Below are the imagined views you would obtain if you took the corresponding actions. "
            "These are provided to help you answer the question.\n"
            "You can include them in your reasoning.\n"
        )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
            
            # Add validated claims if available
            if action_claims and action_str in action_claims and subaction_str in action_claims[action_str]:
                claims = action_claims[action_str][subaction_str]
                if claims:
                    content.append(("Validated observations from this view:\n",))
                    for claim in claims:
                        content.append((f"  - {claim}\n",))
                    content.append(("\n",))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    instructions = (
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content
