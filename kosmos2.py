import os

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForVision2Seq
from time import time
from dataset import CocoDetection
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils


class Kosmos2ObjectDetector:
    def __init__(self):
        model_id  = "microsoft/kosmos-2-patch14-224"
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_id)

    def detect(self, image, prompt="<grounding> Describe the image in detail: "):
        # Convert the OpenCV image (numpy array) to a PIL Image
        # image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        _, entities = self.processor.post_process_generation(generated_text, cleanup_and_extract=True)

        objects = []
        for entity in entities:
            entity_name, (start, end), bbox = entity
            if start == end:
                # skip bounding bbox without a `phrase` associated
                continue
            objects.append([entity_name, bbox])

        return objects


def run_examples(images_path="images"):
    detector = Kosmos2ObjectDetector()
    total = 0
    for img in os.listdir(images_path):
        filename = f"{images_path}/{img}"

        # Load an image using OpenCV
        image = Image.open(filename)  # Replace with your image path
        w, h = image.size
        s = time()
        objects = detector.detect(image, prompt="<grounding> Describe the image in detail:")
        e = time()
        total += e - s
        print(objects)
        print(f"time took for prediction: {e - s}")
        for obj in objects:
            bbox = obj[1][0]
            label = obj[0]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            box = [x1, y1, x2, y2]
            # print(f"Detected {obj[0]} at location {box}")
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline="red", width=3)
            font = ImageFont.load_default()
            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]  # Get text size
            text_background = [x1, y1 - text_height, x1 + text_width, y1]
            draw.rectangle(text_background, fill="red")
            draw.text((x1, y1 - text_height), label, fill="white", font=font)
        image.save(f"resultsPersonCar/{img}")
    avg_time = total / len(os.listdir(f"{images_path}"))
    print(f"avg time is: {avg_time}")


def eval(datapath):
    detector = Kosmos2ObjectDetector()
    prompt = "<grounding> Find vehicles and people. use only the words vehicle and person:"
    ds = CocoDetection(datapath)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    predictions = {}
    for images, targets in tqdm(dl, total=len(dl)):
        images = list(images)
        objects = detector.detect(images, prompt=prompt)
        img_id = targets["image_id"]
        bboxes = []
        labels = []
        for i, obj in enumerate(objects):
            l, bbox = obj
            tag = 1
            # for word in ["man", "human", "person", "woman", "men", "women", "solider", "people", "girl", "boy", "baby"]:
            #     if word in l.lower():
            #         tag = 0
            #         break
            # for word in ["car", "truck", "vehicle", "bike", "automobile", "jeep", "offroad", "off road"]:
            #     if word in l.lower():
            #         tag = 1
            #         break
            if tag not in [0, 1]:
                continue
            image = images[0]
            w, h = image.shape[2], image.shape[1]
            bbox = bbox[0]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            box = [x1, y1, x2, y2]
            bboxes.append(box)
            labels.append(tag)
        scores = [1] * len(bboxes)
        predictions[img_id.item()] = {"boxes": bboxes, "labels": labels, "scores": scores}
    coco_predictions = utils.prepare_for_coco_detection(predictions)
    utils.evaluate(coco_predictions, dl)


if __name__ == "__main__":
    # run_examples(images_path="PersonCar/train/images")
    eval("PersonCar/train")
