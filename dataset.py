import torchvision
import os
from torchvision import transforms


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, data_folder):
        ann_file = os.path.join(data_folder, "annotations.json")
        img_folder = os.path.join(data_folder, "images")
        # ann_file = f"{img_folder}/annotations.json"
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.mytransforms = transforms.Compose([transforms.PILToTensor()]) #, transforms.Resize((640, 640))])

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        img = self.mytransforms(img)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        return img, target