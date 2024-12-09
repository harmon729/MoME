import re
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from models.mome_model import MoMEModel
from preprocessors.multi_image_processor import MultiImageProcessor

model = MoMEModel.from_config("configs/mome.yaml").to("cuda:0")
image_processor = MultiImageProcessor()
colors = ['#0077BB', '#EE7733', '#009988', '#EE3377']
middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

def box_extract(string: str):
    ret = []
    for bboxes_str in middle_brackets_pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret

def extract_ans(string: str):
    try:
        list_of_boxes = box_extract(string)
        if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
            return None
        box = list_of_boxes[0][0]
        if len(box) != 4:
            return None
        return box
    except Exception as e:
        print(f"extract_ans for {string} but get exception: {e}")
        return None

def get_image_from_plt(plt_image):
    plt_image.canvas.draw()
    buf = plt_image.canvas.tostring_rgb()
    width, height = plt_image.canvas.get_width_height()
    return Image.frombytes("RGB", (width, height), buf)

def draw_mole_outcome(mole_routing_outcome: List[np.ndarray]) -> np.ndarray:
    ratio_l = np.array(mole_routing_outcome)
    layer_indices = np.array([i for i in range(0,32)])

    ratio_l = ratio_l[layer_indices, :]
    n_layers = len(layer_indices)
    bar_width = 0.85
    
    plt.figure(figsize=(8, 1.5))
    bottom = np.zeros(n_layers)
    for i in range(ratio_l.shape[1]):
        plt.bar(np.arange(n_layers)+1, ratio_l[:, i], bar_width, bottom=bottom, color=colors[i], label=f'Expert {i+1}', alpha=0.8)
        bottom += ratio_l[:, i]

    plt.legend(loc='upper center', ncol=4)
    plt.xlim(0, 33)
    plt.ylim(0, 1.8)
    plt.yticks([])
    plt.xticks(layer_indices+1)
    plt.tight_layout()
    
    return get_image_from_plt(plt.gcf())

def draw_move_outcome(move_routing_outcome: np.ndarray) -> plt.Figure:
    data = move_routing_outcome
    data = data / data.sum(axis=1)[:, np.newaxis]

    n_layers = 1
    bar_width = 0.75
    labels = ["CLIP", "DINO", "Pix2Struct"]
    
    plt.figure(figsize=(8, 1.5))

    left = np.zeros(n_layers)
    for i in range(data.shape[1]):
        plt.barh(np.arange(n_layers), data[:, i], bar_width, left=left, color=colors[i], label=f'{labels[i]}', alpha=0.8)
        left += data[:, i]

    plt.ylabel('')
    plt.yticks([])
    plt.xticks(np.linspace(0, 1, 11), ['{:.0%}'.format(x) for x in np.linspace(0, 1, 11)])
    plt.legend(loc='upper center', ncol=3)
    plt.xlim(0, 1.0)
    plt.ylim(-0.7, n_layers + 0.3)

    plt.tight_layout()
    return get_image_from_plt(plt.gcf())

def draw_bbox(image, bbox):
    image = image.copy()
    x1, y1, x2, y2 = [bbox[0]*image.width, bbox[1]*image.height, bbox[2]*image.width, bbox[3]*image.height]
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x1,y1),(x2,y2)], outline=(255, 0, 0), width=5)
    return image

def respond_and_update_images(user_input, user_image):
    user_image = Image.fromarray(user_image).convert("RGB")
    model_inputs = image_processor.prepare_multi_image_input(user_image)
    model_inputs = {k: v.to(model.device).unsqueeze(0) for k, v in model_inputs.items()}
    model_inputs.update({"question": [user_input]})
    
    output = model.generate(model_inputs, max_length=50)
    
    move_routing_outcome = model.move_router.routing_outcome.cpu().detach().numpy()
    mole_routing_outcome = []
    for layer in model.llm_model.model.layers:
        mole_routing_outcome.append(layer.mlp.adapter.routing_outcome.cpu().detach().numpy()[0])
    
    move_visualized = draw_move_outcome(move_routing_outcome)
    mole_visualized = draw_mole_outcome(mole_routing_outcome)
    response = [[user_input, output[0]]]
    bbox = extract_ans(output[0])
    
    if bbox is not None:
        user_image_with_box = draw_bbox(user_image, bbox)
        response.append([None, gr.Image(user_image_with_box)])
    
    return response, move_visualized, mole_visualized


def main():
    with gr.Blocks() as demo:
        gr.Markdown('<h1 align="center">MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models</h1>')
        with gr.Row():
            with gr.Column(scale=1):
                chat_input_image = gr.Image(label="Image")
                chat_input_text = gr.Textbox(label="Instruction", placeholder="Enter the instruction and press Enter", submit_btn=True)
                gr.Examples(
                    examples=[
                        ["In the given image, can you find the blue car right and tell me its coordinates?", plt.imread("./demo_figs/REC.jpg")],
                        ["Which is the micro blogging social site that limits each post to 140 characters?", plt.imread("./demo_figs/pix.png")],
                        ["Please describe this image in short.", plt.imread("./demo_figs/General.jpg")],
                        ["In your own words, what sets the designated area [0.616,0.602,0.987,0.864] in image apart?", plt.imread("./demo_figs/REG.jpg")]
                    ],
                    inputs=[chat_input_text, chat_input_image],
                    label="Examples",
                )
                

            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chat", height=500)
                img1_display = gr.Image(label="MoVE") 
                img2_display = gr.Image(label="MoLE")

        chat_input_text.submit(
            fn=respond_and_update_images, 
            inputs=[chat_input_text, chat_input_image], 
            outputs=[chatbot, img1_display, img2_display]
        )

    demo.launch(server_port=7869)
    
if __name__ == "__main__":
    main()