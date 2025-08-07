import torch
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

from transformers import AutoProcessor, AutoModel, Blip2ForImageTextRetrieval, AddedToken, BitsAndBytesConfig, AutoTokenizer
from utils import SAMProcessor

import argparse
from dataloader import RelDataset

import matplotlib.pyplot as plt
import seaborn as sns

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

        # self.processor.num_query_tokens = self.model.config.num_query_tokens
        # image_token = AddedToken("<image>", normalized=False, special=True)
        # self.processor.tokenizer.add_tokens([image_token], special_tokens=True)

        # self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
        # self.model.config.image_token_index = len(self.processor.tokenizer) - 1

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
        if len(pred) == 0:
            return

        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        pred_triplet = [str(p[0] + " " + p[1] + " " + p[2]) for p in pred]

        text_list = [gt_triplet] + pred_triplet

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


        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
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
       
        #text = [f'This is a photo of a {label}.' for label in text]

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
        if len(pred) == 0:
            return

        # compute score for GT triplet:
        gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

        pred_triplet = [str(p[0] + " " + p[1] + " " + p[2]) for p in pred]

        text_list = [gt_triplet] + pred_triplet

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
        self.clip_score_matching.append(scores[0].item())

        # to list
        scores = scores.cpu().to(torch.float32)  
        return scores

class CLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, model_name="clip-large", device="cuda"):
        super(CLIPScoreMatching, self).__init__()
        self.device = device
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-B-32', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/negCLIP.pt", device=self.device)
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-H-14', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/h14_v1.2_altogether.pt", device=self.device)
        self.model_name = model_name
        if self.model_name == "siglip":
            self.model_id = "google/siglip-so400m-patch14-384"
        elif self.model_name == "clip-large":
            self.model_id = "openai/clip-vit-large-patch14"
            # self.model_id = "Nano1337/negclip"
            # self.model_id = "TripletCLIP/CC12M_TripletCLIP_ViTB12"
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
        if image.mode != "RGB":
            image = image.convert("RGB")

        new_eval = False
        if new_eval:
            cos_scores = torch.zeros((len(text),), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for i, t in enumerate(text):
                    inputs_img = self.processor(images=image, return_tensors="pt").to(self.device)
                    inputs_txt = self.tokenizer(t, padding=True, return_tensors="pt").to(self.device)

                    txt_feat = self.model.get_text_features(**inputs_txt)
                    image_features = self.model.get_image_features(**inputs_img)

                    txt_feat = txt_feat / txt_feat.norm(
                        dim=1, keepdim=True).to(torch.float32)
                    image_features = image_features / image_features.norm(
                        dim=1, keepdim=True).to(torch.float32)

                    # calculate score
                    score = (image_features * txt_feat).sum().unsqueeze(0)
                    cos_scores[i] = score

            cos_scores = cos_scores.unsqueeze(0)
        else:

            with torch.no_grad():
                if self.model_id == "Nano1337/negclip":
                    image = self.preprocess(image).unsqueeze(0)
                    text = self.tokenizer(text)
                    image_features = self.model.encode_image(image)
                    text_features = self.model.encode_text(text)
                else:
                    inputs = self.processor(
                        text=text, images=image, return_tensors="pt", padding="max_length"
                    ).to(self.device)
                    outputs = self.model(**inputs)

                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                cos_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return cos_scores[0].cpu()
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
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

        if scores.argmax() == 0:
            self.true_positives += 1
        else:
            self.false_positives += 1
        self.clip_score_matching.append(scores[0].item())

        # to list
        scores = scores.cpu().to(torch.float32)  
        return scores

class PEScoreMatching(SceneGraphEvaluation):
    def __init__(self, device="cuda"):
        super(PEScoreMatching, self).__init__()
        self.device = device
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        # CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']

        self.model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)  # Downloads from HF
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

        with torch.no_grad(), torch.autocast("cuda"):
            cos_scores = torch.zeros((len(text),), dtype=torch.float32, device=self.device)
            for i, t in enumerate(text):
                te = self.tokenizer(t).cuda()
                image_features, txt_feat, _ = self.model(image, te)

                txt_feat = txt_feat / txt_feat.norm(
                        dim=1, keepdim=True).to(torch.float32)
                image_features = image_features / image_features.norm(
                    dim=1, keepdim=True).to(torch.float32)

                # calculate score
                score = (image_features * txt_feat).sum().unsqueeze(0)
                cos_scores[i] = score

            # cos_scores = cos_scores.unsqueeze(0)

            #image_features, text_features, logit_scale = self.model(image, text)
            #text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        return cos_scores.cpu()
    
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
        self.clip_score_matching.append(scores[0].item())

        return scores.cpu().numpy()

def do_evaluation(dataset_name, max_samples, device):
    dataset = RelDataset(dataset_name, max_samples=max_samples, split='train', negative=False)

    save = True
    save_dir = 'eval_alignment_3'

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluators = [
        CLIPScoreMatching(model_name="clip-large", device=device),
        # SIGLIPScoreMatching(model_name="siglip", device=device),
        # SIGLIPScoreMatching(model_name="siglip", device=device),
        # BLIPScoreMatching(device=device),
        #PEScoreMatching(device=device)
    ] #, BLIPScoreMatching(device=device) # , CLIPScoreMatching(model_name='siglip')
    # evaluator2 = CLIPScoreMatching(model_name="clip-large", device=device)
    # evaluator3 = CLIPScoreMatching(model_name="siglip", device=device)

    #sam_processor = SAMProcessor(device='cuda')

    if save:
        for evaluator in evaluators:
            # create directory if it does not exist
            import os
            dir = save_dir + '/' + evaluator.__class__.__name__
            if not os.path.exists(dir):
                os.makedirs(dir)

    possible_preds = dataset.get_possible_predicates()

    for sample in tqdm(dataset):
        img_id, all_gt, all_cropped, all_imgs, all_boxes, _ = sample
        for i, (gt, img_cropped, img_base, boxes) in enumerate(zip(all_gt, all_cropped, all_imgs, all_boxes)):
            sub_label, rel_label, obj_label = gt

            # print info on the PIL image
            #img_with_masks = sam_processor.generate_masks(img_base, boxes, cropping=True, annotate=True)

            # sub_label = sub_label.split("_")[-1]
            # obj_label = obj_label.split("_")[-1]

            sub_label_f = sub_label +'_1'
            obj_label_f = obj_label +'_2'

            if possible_preds[(sub_label, obj_label)] != [rel_label]:
                preds = possible_preds[(sub_label, obj_label)]
                if rel_label in preds:
                    preds.remove(rel_label)
                all_triplets = [(sub_label_f, p, obj_label_f) for p in preds]
                gt_pred = (sub_label_f, rel_label, obj_label_f)
            else:
                continue
            
            for evaluator in evaluators:
                scores = evaluator.calculate(gt_pred, all_triplets, img_base)

                if save:
                    if i == 1:
                        all_pred = [gt_pred[1]]
                        all_pred = all_pred + [p[1] for p in all_triplets]
                        assert len(all_pred) == len(scores), "Scores and predictions length mismatch"

                        # sort the scores and predicates by score
                        sorted_indices = scores.argsort()
                        scores = scores[sorted_indices]
                        all_pred = np.array(all_pred)[sorted_indices]

                        # display a chart which represents the scores for each predicate
                        # the x-axis is the predicate, the y-axis is the score
                        sns.set_theme(style="whitegrid")
                        plt.figure(figsize=(10, 5))
                        # combine the chart with the image cropped img to form a single image, the cropped img is on the left and the chart is on the right, keeping aspect ratio of img_cropped
                        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                        ax[0].imshow(img_cropped)
                        ax[0].axis('off')
                        
                        ax[1].plot(all_pred, scores, color='blue')
                        # add the score for each point, show also the point 'o'
                        for j, score in enumerate(scores):
                            # make the text rotated 75 deg
                            ax[1].text(j, score, f"{score:.2f}", fontsize=10, rotation=75, ha='center', va='bottom')
                        ax[1].scatter(range(len(all_pred)), scores, color='blue', marker='o')

                        ax[1].set_xlabel('Predicate')
                        ax[1].set_ylabel('Score')
                        # ax[1].set_title('GT: ' + gt_pred[0] + ' ' + gt_pred[1] + ' ' + gt_pred[2] + ' | Rank: ' + str(np.where(all_pred == gt_pred[1])[0][0] + 1) + ' / ' + str(len(all_pred)))
                        ax[1].set_xticks(range(len(all_pred)))
                        ax[1].set_xticklabels(all_pred, rotation=75)
                        plt.tight_layout()
                        d = save_dir + '/' + evaluator.__class__.__name__
                        plt.savefig(f"{d}/Img_{img_id}.png")
                        plt.close()

    for evaluator in evaluators:
        print(evaluator.generate_print_string())
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PSG', help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()

    do_evaluation(args.dataset, args.max_samples, args.device)

if __name__ == '__main__':
    main()