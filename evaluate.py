import torch
import numpy as np
import json
from tqdm import tqdm

from abc import ABC, abstractmethod

from collections import Counter
from transformers import AutoProcessor, AutoModel

from models import GPT4Model, Phi3Model, LlamaModel, LLaVAModel, PaliGemmaModel, Qwen2VLModel

import argparse
from dataloader import RelDataset

import time

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
    def __init__(self):
        from lavis.models import load_model_and_preprocess
        from lavis.processors import load_processor
        super(BLIPScoreMatching, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.device, is_eval=True)

        self.recall = []
        self.precision = []

        self.blip_score_matching = []
        self.blip_itm_matching = []
        self.blip_gt_match_cos = []

    def generate_print_string(self):
        match_cos = str(self.blip_gt_match_cos.count(1)) + " / " + str(len(self.blip_gt_match_cos))
        result_str = 'SGG eval: '
        result_str += '    BLIP ITM Score Matching: %.4f; ' % np.mean(self.blip_score_matching)
        result_str += '    BLIP RECALL: %.4f; ' % np.mean(self.recall)
        # result_str += '    BLIP NUMBER OF GT MATCHES COSINE: %s; ' % match_cos
        result_str += '\n'

        return result_str
    
    def preprocess_img(self, image):
        img = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        return img
    
    def compute_similarity_batch(self, text, image, method='itm'):
        if type(text) == str:
            text = [text]
        txt = []

        for t in text:
            #t = "a photo of a " + t
            txt.append(self.text_processors["eval"](t))
        images = torch.stack([image for item in txt]).squeeze(1).to(self.device)
        
        if method == 'cosine':
            with torch.no_grad():
                score = self.model({"image": images, "text_input": txt}, match_head='itc')
                score = score.cpu()
        elif method == 'itm':
            with torch.no_grad():
                itm_output = self.model({"image": images, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            score = itm_scores[:, 1].cpu()
        else:
            print("Not supported method")
            return None
        return score
    
    def compute_similarity(self, text, image, method='itm'):
        scores = []
        for t in text:
            #t = "a photo of a " + t
            txt = self.text_processors["eval"](t)
        
            if method == 'cosine':
                with torch.no_grad():
                    score = self.model({"image": image, "text_input": txt}, match_head='itc')
                    score = score.cpu()
                    scores.append(score)
            elif method == 'itm':
                with torch.no_grad():
                    itm_output = self.model({"image": image, "text_input": txt}, match_head="itm")
                    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                score = itm_scores[:, 1].cpu()
                scores.append(score)
            else:
                print("Not supported method")
                return None
        return scores
    
    def calculate(self, preds, groundtruths, images):
        true_positives = 0
        false_positives = 0
        
        if len(preds) == 0 and len(groundtruths) != 0:
            self.recall.append(0)
            self.precision.append(0)
            return

        for pred, gt, union_img in zip(preds, groundtruths, images):
            # PIL to tensor
            image_features = self.preprocess_img(union_img)

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

class CLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, model_name="siglip"):
        super(CLIPScoreMatching, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-B-32', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/negCLIP.pt", device=self.device)
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-H-14', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/h14_v1.2_altogether.pt", device=self.device)

        if model_name == "siglip":
            model_name = "google/siglip-so400m-patch14-384"
        elif model_name == "clip-large":
            model_name = "openai/clip-vit-large-patch14"
        elif model_name == "clip-base":
            model_name = "openai/clip-vit-base-patch32"

        # model_name = "google/siglip-so400m-patch14-384" # "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"
        # model_name = "openai/clip-vit-large-patch14"

        self.processor = AutoProcessor.from_pretrained(
            model_name,
        )
        torch_dtype = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            device_map=self.device,
            torch_dtype=torch_dtype,
        )

        self.recall = []
        self.precision = []

        self.clip_score_matching = []
        self.refclip_score_matching = []

    def generate_print_string(self):
        result_str = 'SGG eval: '
        result_str += '    CLIP Score Matching: %.4f; ' % np.mean(self.clip_score_matching)
        result_str += '    REFCLIP Score Matching: %.4f; ' % np.mean(self.refclip_score_matching)
        result_str += '    CLIP RECALL: %.4f; ' % np.mean(self.recall)
        result_str += '\n'

        return result_str
    
    def compute_similarity(self, text, image):
        if type(text) == str:
            text = [text]
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            cos_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return cos_scores[0].cpu()
    
    def compute_similarity_text(self, text1, text2):
        inputs = self.processor(
            text=text1, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

            text_features1 = outputs
            text_features1 /= text_features1.norm(dim=-1, keepdim=True)

        inputs = self.processor(
            text=text2, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

            text_features2 = outputs
            text_features2 /= text_features2.norm(dim=-1, keepdim=True)

        cos_scores = (100.0 * text_features1 @ text_features2.T).softmax(dim=-1)

        return cos_scores[0].cpu()
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
    def calculate(self, preds, groundtruths, images):
        true_positives = 0
        false_positives = 0
        
        if len(preds) == 0 and len(groundtruths) != 0:
            self.recall.append(0)
            self.precision.append(0)
            return

        for pred, gt, union_img in zip(preds, groundtruths, images):
            # compute score for GT triplet:
            gt_triplet = str(gt[0] + " " + gt[1] + " " + gt[2])

            pred_triplet = str(pred[0] + " " + pred[1] + " " + pred[2])

            text_list = [gt_triplet, pred_triplet]

            scores = self.compute_similarity(text_list, union_img)
            
            if scores[1] >= scores[0]:
                true_positives += 1
            else:
                false_positives += 1
            self.clip_score_matching.append(scores[1].item())

            scores_text = self.compute_similarity_text(gt_triplet, pred_triplet)

            refclipscores = 2 * scores[1] * scores_text[0] / (scores[1] + scores_text[0])

            self.refclip_score_matching.append(refclipscores.item())

        # compute recall and precision
        recall = true_positives / len(preds)
        precision = true_positives / (true_positives + false_positives)

        self.recall.append(recall)
        self.precision.append(precision)

def do_evaluation(dataset_name, model_name, max_samples, device, save=False):
    dataset = RelDataset(dataset_name, max_samples=max_samples, split='test', negative=False)

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluator = BLIPScoreMatching()
    evaluator2 = CLIPScoreMatching()

    if model_name == 'phi3':
        model = Phi3Model(device=device)
    elif model_name == 'llama':
        model = LlamaModel(device=device)
    elif model_name == 'llava':
        model = LLaVAModel(device=device)
    elif model_name == 'gpt4':
        model = GPT4Model(device=device)
    elif model_name == 'gwen2vl':
        model = Qwen2VLModel(device=device)
    elif model_name == 'paligemma':
        model = PaliGemmaModel(device=device)

    if save:
        new_data = dataset.data.copy()
        i=0

    avg_fps = 0
    avg_ms = 0

    for sample in tqdm(dataset):
        img_id, all_gt, all_imgs, all_orig_imgs = sample

        preds = []
        gt_preds = []
        to_remove = []
        j = 0
        for gt, img_cropped, img_base in zip(all_gt, all_imgs, all_orig_imgs):
            sub_label, rel_label, obj_label = gt

            CoT_template = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Follow these refined steps:\n\n1. **Entity Identification**  \n   - Clearly identify each entity's visual characteristics within the image. Note attributes such as shape, size, and position, alongside any distinguishing features.\n\n2. **Spatial Context**  \n   - Analyze the positioning of the entities relative to each other. Consider aspects like distance, orientation, and whether entities overlap or are close in proximity.\n\n3. **Functional Context**  \n   - Assess any potential functional interactions. Identify actions or roles that may indicate how Entity 1 impacts or interacts with Entity 2.\n\n4. **Integrative Reasoning**  \n   - Determine whether the relationship is predominantly spatial or functional. Use the analysis from steps 1-3 to support your reasoning and articulate it in a concise sentence.\n\nFinally, summarize the visual relationship in the format: `<sub>Entity 1</sub> <rel>relationship</rel> <obj>Entity 2</obj>`. Example: `<sub>1_person</sub> <rel>holding</rel> <obj>2_phone</obj>`.\n\n# Output Format\n\n- Provide your response as a structured sentence summarizing the relationship, followed by the formatted statement. \n\n# Notes\n\n- Ensure that the reasoning provided is comprehensive and ties together observations from all steps.\n- Consider both tangible interactions and abstract spatial nuances when formulating the result. \n Now, predict <rel></rel> for <sub>"+sub_label+"</sub> and <obj>"+obj_label+"</obj> based on this image."

            t_start = time.time()
            predicate, _ = model.generate((sub_label, obj_label), img_cropped)
            t_end = time.time()
            ms = t_end - t_start
            avg_ms += ms
            avg_fps += 1 / (t_end - t_start)

            sub_label = sub_label.split("_")[-1]
            obj_label = obj_label.split("_")[-1]

            if predicate is not None:
                preds.append((sub_label, predicate, obj_label))
                gt_preds.append((sub_label, rel_label, obj_label))

                if save:
                    new_data[i]['relationships'][j]['predicate'] = predicate
            else:
                if save:
                    # remove new_data[i]['relationships'][j]
                    to_remove.append(j)

            j += 1
        
        if save:
            # remove all elements in to_remove
            new_data[i]['relationships'] = [new_data[i]['relationships'][j] for j in range(len(new_data[i]['relationships'])) if j not in to_remove]
            i += 1

        # convert all_imgs from PIL format to json serializable format
        all_imgs = [img.tobytes().decode("latin1") for img in all_imgs]

        if len(gt_preds) == 0:
            continue
        evaluator.calculate(preds, gt_preds, all_orig_imgs)
        evaluator2.calculate(preds, gt_preds, all_orig_imgs)

    print(evaluator.generate_print_string())
    print(evaluator2.generate_print_string())

    avg_fps /= len(dataset)
    avg_ms /= len(dataset)

    print("Average FPS: ", avg_fps)
    print("Average ms: ", avg_ms)

    # save all data
    if save:
        file_name = "output/test_prompt1_"+dataset_name+"_"+model_name+".json"
        with open(file_name, "w") as f:
            json.dump(new_data, f)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PSG', help='Dataset to use')
    parser.add_argument('--model', type=str, default='gpt4', help='Model to use')
    parser.add_argument('--out_path', type=str, default='sampled_data_llava_CoT.json', help='Output file path')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    do_evaluation(args.dataset, args.model, args.max_samples, args.device, save=True)

if __name__ == '__main__':
    main()