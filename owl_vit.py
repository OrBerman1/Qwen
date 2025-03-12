import os

import torch
from PIL import Image, ImageDraw, ImageFont
from time import time
from transformers import OwlViTProcessor, OwlViTForObjectDetection


processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")


queries = [["photo of a car", "photo of a person"]]  # You can change these to the classes you want to find

total = 0
for img in os.listdir("images"):
    image = Image.open(f"images/{img}")

    s = time()
    inputs = processor(text=queries, images=image, return_tensors="pt")

    # Run the model with both image and text inputs
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=queries
    )
    e = time()
    print(f"time took: {e-s}")
    total += e - s

    result = results[0]
    boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
    for box, score, label in zip(boxes, scores, text_labels):
        box = [int(i) for i in box.tolist()]
        x1, y1, _, _ = box
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
        font = ImageFont.load_default()
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]  # Get text size
        text_background = [x1, y1 - text_height, x1 + text_width, y1]
        draw.rectangle(text_background, fill="red")
        draw.text((x1, y1 - text_height), label, fill="white", font=font)
    image.save(f"resultsowl/{img}")

print(f"avg time: {total/len(os.listdir('images'))}")
