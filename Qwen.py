import os
from time import time
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=False)
#
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=False).eval()
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=False)


# avg = 0
# for image in os.listdir("images"):
#     img_path = f"images/{image}"
#     s = time()
#     query = tokenizer.from_list_format([
#         {'image': img_path},
#         {'text': 'please describe the image?'},
#     ])
#     response, history = model.chat(tokenizer, query=query, history=None)
#     e = time()
#     t = e - s
#     avg += t
#     print(img_path)
#     print(response)
#     print(f"time took: {t}, image size is: {cv2.imread(img_path).shape}")
#     print('\n')
#
# print(f"avg time is: {avg/len(os.listdir('images'))}")


avg = 0
for i, image in enumerate(os.listdir("images")):
    img_path = f"images/{image}"
    s = time()
    query = tokenizer.from_list_format([
        {'image': img_path},
        {'text': 'locate people and vehicles in the image.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    e = time()
    t = e - s
    avg += t
    print(img_path)
    print(response)
    print(f"time took: {t}, image size is: {cv2.imread(img_path).shape}")
    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    try:
        image.save(f'results3/{i}.jpg')
    except:
        print("problem")
    print('\n')

print(f"avg time is: {avg/len(os.listdir('images'))}")
