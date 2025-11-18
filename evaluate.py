import torch
import numpy as np
import json
from tqdm import tqdm

from abc import ABC, abstractmethod

from collections import Counter
from transformers import AutoProcessor, AutoModel, Blip2ForImageTextRetrieval, AutoTokenizer

from models import GPT4Model, Phi3Model, LlamaModel, LLaVAModel, PaliGemmaModel, Qwen2VLModel, PEModel, InternVLModel

import argparse
from data.dataloader import RelDataset

import time, os

from sklearn.metrics import precision_score, recall_score, accuracy_score

import torch.nn.functional as F

class SceneGraphEvaluation(ABC):
    def __init__(self):
        super().__init__()
 
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

    def calculate(self, global_container, local_container, mode):
        pass

class SceneGraphEvaluation(ABC):
    def __init__(self):
        super().__init__()
 
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

    def calculate(self, global_container, local_container, mode):
        pass


class BLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, device="cuda"):
        self.device = device

        self.model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

        self.model.to(device)

        self.recall = []
        self.precision = []

        self.blip_score_matching = []
        self.blip_itm_matching = []
        self.blip_gt_match_cos = []

        self.method = 'itm'  # 'itm' or 'cosine'

        self.true_positives = 0
        self.false_positives = 0

        self.new_rel_score = []

    def generate_print_string(self):
        # compute recall and precision
        precision = self.true_positives / (self.true_positives + self.false_positives)
    
        result_str = 'SGG eval: '
        result_str += '    BLIP Score Matching: %.4f; ' % np.mean(self.blip_score_matching)
        result_str += '    BLIP STD: %.4f; ' % np.std(self.blip_score_matching)
        result_str += '    BLIP PRECISION: %.4f; ' % precision
        result_str += '    BLIP NUMBER OF GT MATCHES: %s; ' % str(self.true_positives) + " / " + str(len(self.blip_score_matching))
        result_str += '    BLIP NEW REL SCORE: %.4f; ' % np.mean(self.new_rel_score)
        result_str += '\n'

        return result_str
    
    def compute_similarity(self, text, image, method='itm'):
        scores = []

        if method == 'cosine':
            with torch.no_grad(), torch.autocast("cuda"):
                inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
                itc_out = self.model(**inputs, use_image_text_matching_head=False)
            logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
            scores = logits_per_image.softmax(dim=1)[0]

        elif method == 'itm':
            for t in text:
                t = "a photo of a " + t
                with torch.no_grad(), torch.autocast("cuda"):
                    inputs = self.processor(images=image, text=t, return_tensors="pt").to(self.device)
                    itm_out = self.model(**inputs, use_image_text_matching_head=True)
                logits_per_image = torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
                score = logits_per_image.softmax(dim=1)[0][0]
                # print("Logits: ", logits_per_image)
                # print("Scores: ", score)
                scores.append(score) #logits_per_image[0][0]
        else:
            print("Not supported method")
            return torch.tensor([0.0])
        return torch.tensor(scores)
    
    def calculate(self, preds, images):
        # compute only the blip score matching

        for pred, union_img in zip(preds, images):

            # compute score for GT triplet:
            pred_triplet = str(pred[0] + " " + pred[1] + " " + pred[2])

            text_list = [pred_triplet]

            scores = self.compute_similarity(text_list, union_img, method=self.method)
            
            self.blip_score_matching.append(scores[0].item())
    
    def calculate_ref(self, preds, groundtruths, images):
        true_positives = 0
        false_positives = 0
        
        if len(preds) == 0 and len(groundtruths) != 0:
            self.recall.append(0)
            self.precision.append(0)
            return

        for pred, gt, union_img in zip(preds, groundtruths, images):
            # PIL to tensor
            image_features = self.preprocess_img(union_img)
            if image_features is None:
                self.recall.append(0)
                self.precision.append(0)
                return

            # compute score for GT triplet:
            gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

            pred_triplet = str(pred[0] + " " + pred[1] + " " + pred[2])

            text_list = [gt_triplet, pred_triplet]

            scores = self.compute_similarity(text_list, image_features, method='itm')
            
            if scores[1] >= scores[0]:
                true_positives += 1
            else:
                false_positives += 1
            self.blip_score_matching.append(scores[1].item())

        # compute recall and precision
        recall = true_positives / len(preds)
        precision = true_positives / (true_positives + false_positives)

        self.recall.append(recall)
        self.precision.append(precision)
    
    def calculate(self, gt, pred, union_img):
        

        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        if pred is not None:
            # if there are predictions, compute the score for the GT triplet and the first prediction
            pred_triplet = [str(p[0] + " " + p[1] + " " + p[2]) for p in pred]
            text_list = [gt_triplet] + pred_triplet
        else:
            # if there are no predictions, compute the score for the GT triplet only
            text_list = [gt_triplet]

        scores = self.compute_similarity(text_list, union_img, method=self.method)

        # get rank of gt triplet
        _, indices = torch.sort(scores, descending=True)
        gt_index = indices[0].item()  # index of the GT triplet in the sorted scores
        rel_score = 1-(gt_index / len(scores))  # relative score of the GT triplet
        self.new_rel_score.append(rel_score)

        if scores.argmax() == 0:
            self.true_positives += 1
        else:
            self.false_positives += 1
        
        if pred is not None:
            # if there are predictions, append the score of the first prediction
            self.blip_score_matching.append(scores[1].item())
        else:
            # if there are no predictions, append the score of the GT triplet
            self.blip_score_matching.append(scores[0].item())

        return scores
    
class SIGLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, device="cuda", model_name="siglip2"):
        super(SIGLIPScoreMatching, self).__init__()
        self.device = device

        self.model_name = model_name
        if self.model_name == "siglip":
            self.model_id = "google/siglip-so400m-patch14-384"
        elif self.model_name == "siglip2":
            self.model_id = "google/siglip2-so400m-patch14-384"


        self.model = AutoModel.from_pretrained(self.model_id, device_map="auto",  attn_implementation="sdpa").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.recall = []
        self.precision = []

        self.clip_score_matching = []
        self.refclip_score_matching = []

        self.true_positives = 0
        self.false_positives = 0

        self.new_rel_score = []

    def generate_print_string(self):
        # compute recall and precision
        precision = self.true_positives / (self.true_positives + self.false_positives)
    
        result_str = 'SGG eval: '
        result_str += '    SIGLIP Score Matching: %.4f; ' % np.mean(self.clip_score_matching)
        result_str += '    SIGLIP STD: %.4f; ' % np.std(self.clip_score_matching)
        result_str += '    SIGLIP PRECISION: %.4f; ' % precision
        result_str += '    SIGLIP NUMBER OF GT MATCHES: %s; ' % str(self.true_positives) + " / " + str(len(self.clip_score_matching))
        result_str += '    SIGLIP NEW REL SCORE: %.4f; ' % np.mean(self.new_rel_score)
        result_str += '\n'

        return result_str
    
    def compute_similarity(self, text, image):
        if type(text) == str:
            text = [text]
        # for images in greyscale
        if image.mode != "RGB":
            image = image.convert("RGB")
       
        # texts = [f'This is a photo of a {label}.' for label in text] <-- this does not significantly improve results, even though it should???

        inputs = self.processor(text=text, images=image, padding="max_length", max_length=64, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)

        return probs[0].cpu()
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
    def calculate(self, gt, pred, union_img):
        # compute score for GT triplet:
        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        if pred is not None:
            if not isinstance(pred, list):
                pred = [pred]
            
            pred_triplet = [str(p[0] + " " + p[1] + " " + p[2]) for p in pred]

            text_list = [gt_triplet] + pred_triplet

            scores = self.compute_similarity(text_list, union_img)
        else:
            text_list = [gt_triplet]
            scores = self.compute_similarity(text_list, union_img)

        # normalize the scores between 0 and 1
        # scores = (scores - scores.min()) / (scores.max() - scores.min())

        # if scores[0] - max(scores[1:]) < 0.1:

        # get rank of gt triplet
        _, indices = torch.sort(scores, descending=True)
        gt_index = indices[0].item()  # index of the GT triplet in the sorted scores
        rel_score = 1-(gt_index / len(scores))  # relative score of the GT triplet
        self.new_rel_score.append(rel_score)
        
        if scores.argmax() == 0:
            self.true_positives += 1
        else:
            self.false_positives += 1

        if pred is not None:
            # if there are predictions, append the score of the first prediction
            self.clip_score_matching.append(scores[1].item())
        else:
            # if there are no predictions, append the score of the GT triplet
            self.clip_score_matching.append(scores[0].item())

        # to list
        scores = scores.cpu().to(torch.float32)  
        return scores

class CLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, model_name="clip-large", device="cuda"):
        super(CLIPScoreMatching, self).__init__()
        self.device = device
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-B-32', pretrained="negCLIP.pt", device=self.device)
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-H-14', pretrained="h14_v1.2_altogether.pt", device=self.device)
        self.model_name = model_name
        if self.model_name == "siglip":
            self.model_id = "google/siglip-so400m-patch14-384"
        elif self.model_name == "clip-large":
            self.model_id = "openai/clip-vit-large-patch14" #"Nano1337/negclip"
        elif self.model_name == "clip-base":
            self.model_id = "openai/clip-vit-base-patch32"
        elif self.model_name == "negclip":
            self.model_id = "Nano1337/negclip"

        if self.model_id == "Nano1337/negclip":
            # use open_clip to load the model
            import open_clip

            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:'+self.model_id)
            self.tokenizer = open_clip.get_tokenizer('hf-hub:'+self.model_id)

        else:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
            )
            torch_dtype = torch.bfloat16

            self.model = AutoModel.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                device_map=self.device,
                torch_dtype=torch_dtype,
                weights_only=False,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
            )

        self.recall = []
        self.precision = []

        self.clip_score_matching = []
        self.refclip_score_matching = []

        self.true_positives = 0
        self.false_positives = 0

        self.new_rel_score = []

    def generate_print_string(self):
        # compute recall and precision
        precision = self.true_positives / (self.true_positives + self.false_positives)
    
        result_str = 'SGG eval: '
        result_str += '    CLIP Score Matching: %.4f; ' % np.mean(self.clip_score_matching)
        result_str += '    CLIP REFCLIP Score Matching: %.4f; ' % np.mean(self.refclip_score_matching)
        result_str += '    CLIP STD: %.4f; ' % np.std(self.clip_score_matching)
        result_str += '    CLIP PRECISION: %.4f; ' % precision
        result_str += '    CLIP NUMBER OF GT MATCHES: %s; ' % str(self.true_positives) + " / " + str(len(self.clip_score_matching))
        result_str += '    CLIP NEW REL SCORE: %.4f; ' % np.mean(self.new_rel_score)
        result_str += '\n'

        return result_str
    
    def compute_similarity(self, text, image):
        if type(text) == str:
            text = [text]
        # for images in greyscale
        # if image.mode != "RGB":
        #     image = image.convert("RGB")

        cos_scores = torch.zeros((len(text),), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for i, t in enumerate(text):
                if self.model_id == "Nano1337/negclip":
                    img = self.preprocess(image).unsqueeze(0)
                    te = self.tokenizer(t)
                    image_features = self.model.encode_image(img)
                    text_features = self.model.encode_text(te)
                else:
                    inputs = self.processor(
                        text=t, images=image, return_tensors="pt", padding="max_length"
                    ).to(self.device)
                    outputs = self.model(**inputs)

                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # calculate score
                # score = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                score = (image_features * text_features).sum()

                cos_scores[i] = score
        return cos_scores.cpu()
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
    def calculate(self, gt, pred, union_img):
        # if len(pred) == 0 and pred != None:
        #     return

        # compute score for GT triplet:
        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        if pred is not None:
            pred_triplet = str(pred[0] + " " + pred[1] + " " + pred[2])
            text_list = [gt_triplet, pred_triplet]
            scores = self.compute_similarity(text_list, union_img)
        else:
            scores = self.compute_similarity(gt_triplet, union_img)
        
        # get rank of gt triplet
        _, indices = torch.sort(scores, descending=True)
        gt_index = indices[0].item()  # index of the GT triplet in the sorted scores
        rel_score = 1-(gt_index / len(scores))  # relative score of the GT triplet
        self.new_rel_score.append(rel_score)

        if scores.argmax() == 0:
            self.true_positives += 1
        else:
            self.false_positives += 1

        if pred is not None:
            # compute refclip score
            # refclip score is the similarity between the GT triplet and the predicted triplet
            scores_text = self.compute_similarity_text(gt_triplet, pred_triplet, union_img)

            refclipscores = 2 * scores[0] * scores_text[0] / (scores[0] + scores_text[0])

            self.refclip_score_matching.append(refclipscores.item())
            self.clip_score_matching.append(scores[1].item())
        else:
            self.clip_score_matching.append(scores[0].item())

        scores = scores.cpu().to(torch.float32)  
        return scores
    
    def compute_similarity_text(self, text1, text2, image=None):
        with torch.no_grad():
            if self.model_id == "Nano1337/negclip":
                te = self.tokenizer(text1)
                text_features1 = self.model.encode_text(te)
                te = self.tokenizer(text2)
                text_features2 = self.model.encode_text(te)

            else:
                inputs = self.processor(
                    text=text1, images=image, return_tensors="pt", padding="max_length"
                ).to(self.device)
                outputs = self.model(**inputs)

                text_features1 = outputs.text_embeds

                inputs = self.processor(
                    text=text2, images=image, return_tensors="pt", padding="max_length"
                ).to(self.device)
                outputs = self.model(**inputs)

                text_features2 = outputs.text_embeds

            text_features1 /= text_features1.norm(dim=-1, keepdim=True)
            text_features2 /= text_features2.norm(dim=-1, keepdim=True)
            #     score = (image_features * text_features).sum().unsqueeze(0)
            #     cos_scores[i] = score

        score = 100 * (text_features1 * text_features2).sum(axis=-1)

        return score.cpu()

class PEScoreMatching(SceneGraphEvaluation):
    def __init__(self, device="cuda"):
        super(PEScoreMatching, self).__init__()
        self.device = device
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        # CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

        self.model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        self.model = self.model.cuda()

        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

        self.recall = []
        self.precision = []

        self.clip_score_matching = []

        self.true_positives = 0
        self.false_positives = 0

        self.new_rel_score = []

    def generate_print_string(self):
        # compute recall and precision
        precision = self.true_positives / (self.true_positives + self.false_positives)

        result_str = 'SGG eval: '
        result_str += '    PE Score Matching: %.4f; ' % np.mean(self.clip_score_matching)
        # show std deviation for clip_score_matching
        result_str += '    PE STD: %.4f; ' % np.std(self.clip_score_matching)
        result_str += '    PE PRECISION: %.4f; ' % np.mean(precision)
        result_str += '    PE NUMBER OF GT MATCHES: %s; ' % str(self.true_positives) + " / " + str(len(self.clip_score_matching))
        result_str += '    PE NEW REL SCORE: %.4f; ' % np.mean(self.new_rel_score)
        result_str += '\n'

        return result_str

    def compute_similarity(self, text, image):
        image = self.preprocess(image).unsqueeze(0).cuda()
        text = self.tokenizer(text).cuda()

        with torch.no_grad(), torch.autocast("cuda"):
            image_features, text_features, logit_scale = self.model(image, text)
            text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        return text_probs[0].cpu()
    
    def calculate(self, gt, pred, union_img):
        
        if len(pred) == 0:
            return

        # compute score for GT triplet:
        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        pred_triplet = [str(p[0] + " " + p[1] + " " + p[2]) for p in pred]

        text_list = [gt_triplet] + pred_triplet

        scores = self.compute_similarity(text_list, union_img)

        # get rank of gt triplet
        _, indices = torch.sort(scores, descending=True)
        gt_index = indices[0].item()  # index of the GT triplet in the sorted scores
        rel_score = 1-(gt_index / len(scores))  # relative score of the GT triplet
        self.new_rel_score.append(rel_score)
        
        if scores.argmax().item() == 0:
            self.true_positives += 1
        else:
            self.false_positives += 1
        self.clip_score_matching.append(scores[1].item())

        return scores.cpu().numpy()

def do_evaluation(dataset_name, model_name, max_samples, device, eval_only=False, save=False):
    dataset = RelDataset(dataset_name, max_samples=max_samples, split='train', negative=False, masks=False)

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluators = [CLIPScoreMatching(model_name='negclip', device=device), 
                   SIGLIPScoreMatching(model_name="siglip", device=device),
                CLIPScoreMatching(model_name='clip-large', device=device),
                # BLIPScoreMatching(device=device),
                # SIGLIPScoreMatching(model_name="siglip", device=device)
            ] #PEScoreMatching(device=device), 
    #, CLIPScoreMatching(model_name="clip-large", device=device)
    # evaluator2 = CLIPScoreMatching(model_name="clip-large", device=device)
    # evaluator3 = CLIPScoreMatching(model_name="siglip", device=device)

    if model_name != '':
        if model_name == 'phi3':
            model = Phi3Model(device=device)
        elif model_name == 'llama':
            model = LlamaModel(device=device)
        elif model_name == 'llava':
            model = LLaVAModel(device=device)
        elif model_name == 'gpt4':
            model = GPT4Model(model_id = 'gpt-4o-mini', device=device)
        elif model_name == 'gwen2vl':
            model = Qwen2VLModel(device=device)
        elif model_name == 'paligemma':
            model = PaliGemmaModel(device=device)
        elif model_name == 'perception_encoder':
            model = PEModel(device=device)
        elif model_name == 'internvl':
            model = InternVLModel(device=device)

    if save:
        new_data = dataset.data.copy()
        i=0

    avg_fps = 0
    avg_ms = 0

    recall = []

    eval_ratios = []

    for sample in tqdm(dataset):
        img_id, all_gt, all_cropped, all_imgs, all_boxes, all_ratios = sample

        if len(all_imgs) == 0:
            # print("Skipping image with no rels")
            continue

        # if all_orig_imgs[0] is an image with number of dim != 3, skip
        if len(all_imgs[0].getbands()) != 3:
            print("Skipping image with number of channels != 3")
            if save: 
                new_data[i]['relationships'] = []
                i += 1
            continue

        preds = []
        gt_preds = []
        to_remove = []
        
        j = 0
        for gt, img_cropped, img_base, ratios in zip(all_gt, all_cropped, all_imgs, all_ratios):
            sub_label, rel_label, obj_label = gt

            t_start = time.time()

            if model_name != '':
                predicate, _ = model.generate((sub_label, obj_label), img_cropped)
            else:
                predicate = rel_label
            t_end = time.time()

            ms = t_end - t_start
            avg_ms += ms
            avg_fps += 1 / 1 #(t_end - t_start)

            sub_label = sub_label.split("_")[-1]
            obj_label = obj_label.split("_")[-1]

            if predicate is not None:
                for evaluator in evaluators:
                    if not eval_only:
                        scores = evaluator.calculate((sub_label, rel_label, obj_label), (sub_label, predicate, obj_label), img_base)
                        eval_ratios.append((ratios, scores[0].item()))
                    else:
                        scores = evaluator.calculate((sub_label, predicate, obj_label), None, img_base)
                        eval_ratios.append((ratios, scores[0].item()))
                preds.append((sub_label, predicate, obj_label))
                gt_preds.append((sub_label, rel_label, obj_label))

                if save:
                    new_data[i]['relationships'][j]['predicate'] = predicate
            else:
                if save:
                    # remove new_data[i]['relationships'][j]
                    to_remove.append(j)

            j += 1
        
        # if p >= 100:
        #     print("Skipping image with too many relationships")
        #     break

        if save:
            # remove all elements in to_remove
            new_data[i]['relationships'] = [new_data[i]['relationships'][j] for j in range(len(new_data[i]['relationships'])) if j not in to_remove]
            i += 1

        # convert all_imgs from PIL format to json serializable format
        #all_imgs = [img.tobytes().decode("latin1") for img in all_imgs]

        if len(gt_preds) == 0:
            continue

        gt_preds = [gt[1] for gt in gt_preds]
        preds = [pred[1] for pred in preds]
        recall.append(accuracy_score(gt_preds, preds))

    for evaluator in evaluators:
        print(evaluator.generate_print_string())

    # print overall recall and precision
    print("Overall Accuracy: ", np.mean(recall))

    avg_fps /= len(dataset)
    avg_ms /= len(dataset)

    print("Average FPS: ", avg_fps)
    print("Average ms: ", avg_ms)

    if not os.path.exists("evaluation/"+model_name):
        os.makedirs("evaluation/"+model_name)
    # save all data
    if save:
        file_name = "evaluation/"+model_name+"/test_prompt1_"+dataset_name+"_"+model_name+".json"
        with open(file_name, "w") as f:
            json.dump(new_data, f)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PSG', help='Dataset to use')
    parser.add_argument('--model', type=str, default='', help='Model to use')
    parser.add_argument('--out_path', type=str, default='sampled_data_llava_CoT.json', help='Output file path')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--eval_only', type=str, default=False, help='Eval only flag')
    args = parser.parse_args()

    do_evaluation(args.dataset, args.model, args.max_samples, args.device, args.eval_only, save=False)

if __name__ == '__main__':
    main()