vg_path = "datasets/vg150_full.json"
vg_img_path = "datasets/VG_100K/"
coco_path = "datasets/coco_train_llava_full.json" # "all_annotations/COCO (pseudo)/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json"

coco_path = "datasets/RLIPv2_train2017_threshold20_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05.json"
# coco_path = "generated_data/llava_coco_test.json"
coco_img_path = "datasets/coco/"

o365_path = "datasets/O365 (pseudo)/RLIPv2_o365trainval_Tagger2_Noi24_20e_Xattnmask_SceneGraph_model_large_caption_nucleus10_thre05_1234_4.json"
o365_path = "datasets/obj365_val_llava.json"
o365_img_path = "datasets/object365/images/"
obj365_filenames = "datasets/image_id_to_filepath.json"

psg_path = "datasets/psg_full.json"
PATH_CATALOG = {
    "VG": {'images': vg_img_path, 'data': vg_path},
    "COCO": {'images': coco_img_path, 'data': coco_path},
    "OBJ365": {'images': o365_img_path, 'data': o365_path, 'filenames': obj365_filenames},
    "PSG": {'images': coco_img_path, 'data': psg_path},
    "VG150": {'images': vg_img_path, 'data': vg_path}
}

import numpy as np
from tqdm import tqdm
import json, os
import random
from PIL import Image
import cv2

from sam import SAMProcessor

from torch.utils.data import Dataset

# compute intersection of two bounding boxes
def intersection(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y

    # return boolean
    return w > 0 and h > 0

class RelDataset(Dataset):
    def __init__(self, dataset='COCO', max_samples=1000, split='train', negative=False, masks=False):
        self.dataset_name = dataset
        self.max_samples = max_samples
        assert split in ['train', 'val', 'test'], "Invalid split"
        assert self.dataset_name in PATH_CATALOG.keys(), "Dataset not found"
        self.data, self.images_list = self.load_data(split=split)
        self.negative = negative

        self.masks = masks
        if self.masks:
            self.sam_processor = SAMProcessor(device='cuda')

    def load_data(self, split='train'):
        data_path = PATH_CATALOG[self.dataset_name]['data']
        with open(data_path, 'r') as f:
            print(f"Loading data from {data_path}...")
            json_data = json.load(f)

        data = []

        images_path = PATH_CATALOG[self.dataset_name]['images']
        # create a list of all absolute image paths
        images_list = []
        if self.dataset_name == "OBJ365":
            with open(PATH_CATALOG[self.dataset_name]['filenames'], 'r') as f:
                obj365_filenames = json.load(f)
            data_path = os.path.join(images_path, split)
            for d in tqdm(json_data):
                if d['data_split'] == split:
                    data.append(d)
                    img_id = d['image_id']

                    img_path = obj365_filenames[str(d['image_id'])]
                    img_path = img_path.split('/')[-1]
                    images_list.append({'img_id': img_id, 'img_path' :os.path.join(data_path, img_path)})
        elif self.dataset_name == "VG":
            for d in json_data:
                if d['data_split'] == split:
                    data.append(d)
                    img_id = d['image_id']
                    img_path = os.path.join(images_path, str(img_id) + '.jpg')
                    images_list.append({'img_id': img_id, 'img_path' : img_path})
        elif self.dataset_name == "COCO" or self.dataset_name == "PSG":
            if split == 'train':
                split = 'train2017'
            elif split == 'val':
                split = 'val2017'
            elif split == 'test':
                split = 'test2017'

            if self.dataset_name == "PSG" and split == 'test2017':
                split = 'val2017'
            data_path = os.path.join(images_path, split)
            for d in json_data:
                if d['data_split'] == split:
                    data.append(d)

                    #get the file name in data_path which contains the image_id
                    # fill with zeros in front of the id until 12
                    img_id = str(d['image_id']).zfill(12)
                    img_path = img_id + '.jpg'

                    images_list.append({'img_id': d['image_id'], 'img_path' :os.path.join(data_path, img_path)})

        # sort by d['image_id']
        data = sorted(data, key=lambda x: x['image_id'])
        images_list = sorted(images_list, key=lambda x: x['img_id'])
        self.all_data = data

        data = data[:self.max_samples]
        images_list = images_list[:self.max_samples]
        assert len(data) == len(images_list), "Data and images list have different lengths, got {} and {}".format(len(data), len(images_list))
        return data, images_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.images_list[idx]['img_path']
        img_id = self.images_list[idx]['img_id']
        img = Image.open(img_path)
        img_data = self.data[idx]

        all_gt = []
        all_cropped_imgs = []
        all_orig_imgs = []
        all_boxes = []

        all_ratios = []

        objects = {obj['object_id']: obj for obj in img_data['objects']}

        for rel in img_data['relationships']:
            cropped_img = img.copy()
            orig_img = img.copy()

            sub = objects[rel['subject_id']]
            obj = objects[rel['object_id']]

            # crop the image to the union of the two boxes
            x1, y1, w1, h1 = [sub['x'], sub['y'], sub['w'], sub['h']]
            x2, y2, w2, h2 = [obj['x'], obj['y'], obj['w'], obj['h']]
            # to int
            x1, y1, w1, h1 = map(int, [x1, y1, w1, h1])
            x2, y2, w2, h2 = map(int, [x2, y2, w2, h2])
            
            sub['bbox'] = [x1, y1, w1, h1]
            obj['bbox'] = [x2, y2, w2, h2]

            # here we check if the boxes intersect
            # if intersection(sub['bbox'], obj['bbox']):
            #     continue
    
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            # if possible, add a few pixels around the crop to include context
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.width, w + padding)
            h = min(img.height, h + padding)

            sub_label = "1" # "1_" + sub['names']
            obj_label = "2" #"2_" + obj['names']

            if self.masks:
                # from PIL to an RGB image of shape (H, W, 3) in range [0, 255].
                cropped_img = np.array(cropped_img.convert("RGB"))

                cropped_img = self.sam_processor.generate_masks(cropped_img, [sub['bbox'], obj['bbox']], colors=[(0, 0, 255), (255, 0, 0)],annotate=False)
                # back to PIL
                cropped_img = Image.fromarray(cropped_img.astype(np.uint8))
            else:
                cropped_img = self.visualize_box(cropped_img, sub, sub_label, (0, 0, 255))
                cropped_img = self.visualize_box(cropped_img, obj, obj_label, (255, 0, 0))

            if self.negative:
                # make all pixels within x1, y1, w1, h1 and x2, y2, w2, h2 black
                pixels = orig_img.load()

                # for box1
                for i in range(x1, x1+w1):
                    for j in range(y1, y1+h1):
                        pixels[i-x, j-y] = (0, 0, 0)

                # for box2
                for i in range(x2, x2+w2):
                    for j in range(y2, y2+h2):
                        pixels[i-x, j-y] = (0, 0, 0)

                orig_img = orig_img.crop((x, y, x+w, y+h))
                cropped_img = cropped_img.crop((x, y, x+w, y+h))

            else:
                orig_img = orig_img.crop((x, y, x+w, y+h))

            cropped_img = cropped_img.crop((x, y, x+w, y+h))

            all_cropped_imgs.append(cropped_img)
            all_orig_imgs.append(orig_img)
            all_gt.append((sub['names'], rel['predicate'], obj['names']))
            all_boxes.append((sub['bbox'], obj['bbox']))

            # compute box ratio 
            sub_bbox = sub['bbox']
            obj_bbox = obj['bbox']
            sub_area = sub_bbox[2] * sub_bbox[3]
            obj_area = obj_bbox[2] * obj_bbox[3]
            if sub_area > 0 and obj_area > 0:
                box_ratio = min(sub_area, obj_area) / max(sub_area, obj_area)
            else:
                box_ratio = 0.0

            all_ratios.append(box_ratio)
        
        return img_id, all_gt, all_cropped_imgs, all_orig_imgs, all_boxes, all_ratios
    
    def get_possible_predicates(self):
        """
        For every different subject, object pair, get all the possible predicates annotated in the dataset
        """
        predicates = {}
        for img in self.all_data:
            objects = {obj['object_id']: obj for obj in img['objects']}

            for rel in img['relationships']:

                sub = objects[rel['subject_id']]
                obj = objects[rel['object_id']]
                sub_label = sub['names']
                obj_label = obj['names']

                pred = rel['predicate']
                if (sub_label, obj_label) not in predicates:
                    predicates[(sub_label, obj_label)] = [pred]
                if pred not in predicates[(sub_label, obj_label)]:
                    predicates[(sub_label, obj_label)].append(pred)
        return predicates

    def get_data(self):
        return self.data, self.images_list

    def get_random_sample(self, intersect=True):
        # get a random sample from the data
        while True:
            img_id = random.randint(0, len(self.data))
            img = self.data[img_id]
            img_path = self.images_list[img_id]['img_path']
            if len(img['relationships']) > 0:
                break
        return self.get_sample(img, img_path, intersect)
    
    def get_sample(self, img, img_path, intersect=True):
        # get all objects pairs that intersect
        objects = []
        for obj in img['objects']:
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            objects.append({'bbox': bbox, 'label': obj['names'], 'id': obj['object_id']})

        if intersect:
            # compute intersection for all pairs
            pairs = []
            for i in range(len(objects)):
                for j in range(len(objects)):
                    if i == j:
                        continue
                    if intersection(objects[i]['bbox'], objects[j]['bbox']):
                        # check if the size of one box is > 5 times the size of the other box
                        if objects[i]['bbox'][2] * objects[i]['bbox'][3] > 5 * objects[j]['bbox'][2] * objects[j]['bbox'][3] or \
                           objects[j]['bbox'][2] * objects[j]['bbox'][3] > 5 * objects[i]['bbox'][2] * objects[i]['bbox'][3]:
                            continue
                        pairs.append((objects[i], objects[j]))
                        pairs.append((objects[j], objects[i]))
        else:
            pairs = [(objects[i], objects[j]) for i in range(len(objects)) for j in range(len(objects)) if i != j]
            pairs.extend([(objects[j], objects[i]) for i in range(len(objects)) for j in range(len(objects)) if i != j])
        
        img = Image.open(img_path)

        return img, pairs
    
    def get_pair_data(self, img, pair, negative=False):
        # get the union of the pair and crop the image
        x1, y1, w1, h1 = pair[0]['bbox']
        # to int
        x1, y1, w1, h1 = map(int, [x1, y1, w1, h1])
        x2, y2, w2, h2 = pair[1]['bbox']
        # to int
        x2, y2, w2, h2 = map(int, [x2, y2, w2, h2])
        img3 = img.copy()
        
        img3 = self.visualize_box(img3, pair[1], "2", (0, 0, 255))

        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        # if possible, add a few pixels around the crop to include context
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.width, w + padding)
        h = min(img.height, h + padding)

        sub_label = "1_" + pair[0]['label']
        obj_label = "2_" + pair[1]['label']
        
        cropped_img = img.copy()

        if negative:
            # make all pixels within x1, y1, w1, h1 and x2, y2, w2, h2 black
            pixels = img3.load()

            # for box1
            for i in range(x1, x1+w1):
                for j in range(y1, y1+h1):
                    pixels[i-x, j-y] = (0, 0, 0)

            img2 = img.copy()

            img2 = self.visualize_box(img2, pair[0], "1", (255, 0, 0))

            pixels2 = img2.load()

            # for box2
            for i in range(x2, x2+w2):
                for j in range(y2, y2+h2):
                    pixels2[i-x, j-y] = (0, 0, 0)

            img_cropped = img3.crop((x, y, x+w, y+h))
            img2 = img2.crop((x, y, x+w, y+h))

            return img2, img_cropped, img, (sub_label, obj_label), (pair[0], pair[1])
        
        if self.masks:
            # from PIL to an RGB image of shape (H, W, 3) in range [0, 255].
            cropped_img = np.array(cropped_img.convert("RGB"))
            cropped_img = self.sam_processor.generate_masks(cropped_img, [pair[0]['bbox'], pair[1]['bbox']], colors=["blue", "red"], annotate=False)
        else:
            cropped_img = self.visualize_box(cropped_img, pair[0], sub_label, (0, 0, 255))
            cropped_img = self.visualize_box(cropped_img, pair[1], obj_label, (255, 0, 0))

        cropped_img = cropped_img.crop((x, y, x+w, y+h))
        img3 = img.copy()
        img3 = img3.crop((x, y, x+w, y+h))

        return img, cropped_img, img3, (sub_label, obj_label), (pair[0], pair[1])
    
    def get_random_pair(self, intersect=True, negative=False):
        while True:
            img, pairs = self.get_random_sample(intersect)
            if len(pairs) > 0:
                break
        pair = random.choice(pairs)
        return self.get_pair_data(img, pair, negative)
    
    def data_sampling(self, pairs, sampling, max_samples=50):
        max_pairs = len(pairs)
        num_pairs = min(int(max_pairs * sampling / 100), max_samples)
        return random.sample(pairs, num_pairs)

    def visualize_box(self, img, box, label, color=(0, 0, 255)):
        # Convert PIL image to NumPy array
        img = np.array(img)
        # Convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x, y, w, h = map(int, box['bbox'])  # Ensure coordinates are integers
        # Draw the rectangle on the image
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        # draw a rectangle for the background of the label
        if label != "":
            font = cv2.FONT_HERSHEY_SIMPLEX
            (label_width, label_height), baseline = cv2.getTextSize(label, font, 0.5, 1)
            center_x, center_y = (x + w // 2, y + h // 2)

            # draw the rectangle at the top left corner of the bounding box
            img = cv2.rectangle(img, (center_x, center_y - label_height), (center_x + label_width, center_y + baseline), color, cv2.FILLED)

            # Add the label to the image in the center of the bounding box
            img = cv2.putText(img, label, (center_x, center_y), font, 0.5, (255, 255, 255), 1)

        # Convert back to PIL image if needed
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img