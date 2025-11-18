from data import RelDataset
from models import Phi3Model, LlamaModel, LLaVAModel, GPT4Model, Qwen2VLModel, InternVLModel

import argparse
import random

from tqdm import tqdm
import json

from evaluate import BLIPScoreMatching, CLIPScoreMatching


def data_sampling(pairs, sampling, max_samples=50):
    max_pairs = len(pairs)
    num_pairs = min(int(max_pairs * sampling / 100), max_samples)
    return random.sample(pairs, num_pairs)

def generate_data(dataset, sampling, intersect, model, max_samples=1000, out_path='sampled_data.json', evaluate=False):
    dataset = RelDataset(dataset, split='train', max_samples=max_samples, masks=False)
    print("Loading data for dataset ", dataset)
    data, img_data = dataset.get_data()

    if evaluate:
        evaluator = CLIPScoreMatching(model_name='negclip') # or BLIPScoreMatching(), SIGLIPScoreMatching()
        evaluator2 = CLIPScoreMatching(model_name='clip-large')

    rel_id = 0
    num_pairs = 0
    if model == 'phi3':
        model = Phi3Model()
    elif model == 'llama':
        model = LlamaModel()
    elif model == 'llava':
        model = LLaVAModel()
    elif model == 'gpt4':
        model = GPT4Model()
    elif model == 'qwen2vl':
        model = Qwen2VLModel()
    elif model == 'internvl':
        model = InternVLModel()

    pbar = tqdm(total=len(data))
    new_data = data.copy()

    for i in range(len(new_data)):
        new_data[i]['relationships'] = []

    for i, (sample, img_d) in enumerate(zip(data, img_data)):
        img, pairs = dataset.get_sample(sample, img_d['img_path'], intersect)
        pairs = data_sampling(pairs, sampling)
        num_pairs += len(pairs)

        for pair in pairs:
            img_neg, img_cropped, img3, labels, bboxes = dataset.get_pair_data(img, pair)

            predicate, _ = model.generate(labels, img_cropped)

            if predicate != None:
                rel_id += 1
                new_data[i]['relationships'].append({
                    'relationship_id': rel_id,
                    'subject_id': pair[0]['id'],
                    'predicate': predicate,
                    'object_id': pair[1]['id'],
                    'confidence': 1.0
                })
                pbar.set_description(f"Rels gen: {rel_id} / {num_pairs}")
                triplet = [labels[0], predicate, labels[1]]

                if evaluate:
                    if evaluator is not None: evaluator.calculate(triplet, None, img3)
                    if evaluator2 is not None: evaluator2.calculate(triplet, None, img3)

        # save every 50 images
        if i % 50 == 0:
            with open(out_path, 'w') as f:
                json.dump(new_data, f)
            if evaluate:
                if evaluator is not None: print(evaluator.generate_print_string())
                if evaluator2 is not None: print(evaluator2.generate_print_string())
        pbar.update(1)

    print("Number of relationships generated:", rel_id)
    if evaluate:
        if evaluator is not None: print(evaluator.generate_print_string())
        if evaluator2 is not None: print(evaluator2.generate_print_string())
    with open(out_path, 'w') as f:
        json.dump(new_data, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COCO', help='Dataset to use')
    parser.add_argument('--sampling', type=int, default=25, help='Number of percent of data to sample')
    parser.add_argument('--intersect', type=bool, default=True, help='Only sample pairs that intersect')
    parser.add_argument('--model', type=str, default='llama', help='Model to use')
    parser.add_argument('--out_path', type=str, default='generated_data/sampled_data_gpt4_obj365.json', help='Output file path')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max number of samples to use')
    args = parser.parse_args()

    generate_data(args.dataset, args.sampling, args.intersect, args.model, args.max_samples, args.out_path)

if __name__ == '__main__':
    main()