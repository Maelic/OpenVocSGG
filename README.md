# [SG2RL 2025] Official code for the paper [Measuring Image-Relation Alignment: Reference-Free Evaluation of VLMs and Synthetic Pre-training for Open-Vocabulary Scene Graph Generation](https://openaccess.thecvf.com/content/ICCV2025W/SG2RL/html/Neau_Measuring_Image-Relation_Alignment_Reference-Free_Evaluation_of_VLMs_and_Synthetic_Pre-training_ICCVW_2025_paper.html) presented at the 3rd Workshop on Scene Graphs and Graph Representation Learning at ICCV 2025.

This paper present three mains contributions: (1) **the RelCLIPScore metric**, (2) **a comprehensive evaluation of VLMs for the task of Relation Prediction** and (3) **the Fine-Grained Open-Vocabulary Scene Graph (FG-OV SG) dataset**.

## 1. RelCLIPScore

### Definition

The RelCLIPScore is a new metric for measuring the accuracy of a Scene Graph Generation model, targetted for Open-Vocabulary approaches. In contrast to traditional metrics such as Recall@k or meanRecall@k, RelCLIPScore is a reference-free metric, which means that you don't need groundtruth annotations to compute it. The inspiration comes from the field of Image Captioning where the CLIPScore is being used for a few years now. The RelCLIPScore is in fact the aggregation of individual CLIPScore obtained for every relation predicted by an SGG model. The CLIPScores are obtained by cropping the region on the union of subject and object bounding boxes (or masks) and then feeding the cropped image and associated predicted relation to a CLIP-liked model to compute the cosine similarity between both embeddings (i.e. distance in the embedding space between image and text representations for the predicted relation). 

To force Open-Vocabulary SGG models to predict diverse and informative relations on every image, we add a penalty based on the maximum amount of possible subject-object pairs given a set of groundtruth (or predicted) bounding boxes (or masks), which leads to: 

<p align="center"><img width="50%" width="1366" height="242" alt="image" src="https://github.com/user-attachments/assets/29f6c6f4-8279-4957-a2ce-b6c6e043ba62" /></p>

where p is 50% of the maximum number of pair in the image (as we assume not every combination can be a true positive, even in open-vocabulary settings).

### Experiments

We measured the performance of multiple CLIP-like models for the task of image-relation alignment using our introduce RelCLIPScore metric on the VG and PSG datasets. Results are provided below, where we can see that the NegCLIP model produce higher quality embeddings with a better alignment with groundtruth annotations on both datasets.

<p align="center"><img width="50%" width="1421" height="687" alt="image" src="https://github.com/user-attachments/assets/4874c1c1-493a-4281-820d-b650d7cc8bca" /></p>

## 2. VLMs for Open-Vocabulary SGG

In recent years VLMs have been used to produce Scene Graphs through different prompting strategies. However, approaches mainly rely on prompting for both object detection and relation prediction at the same time, leading to a low diversity of subject-object pairs in generated data (i.e. the VLMs will always choose to focus on the "easy" or salient pairs). In this work, we propose to evaluate the performance of VLMs in generating region-specific relations when the subject and object are randomly sampled from the distribution of groundtruth bounding boxes annotations. We measured the performance of different open or closed-source VLMs using our introduce RelCLIPScore, as follow:

<p align="center"><img width="75%" width="1434" height="387" alt="image" src="https://github.com/user-attachments/assets/86f535e7-8364-46f8-bded-3d200b998948" /></p>

Additional experiments highlight three different limitations:

1. VLMs struggle in **pair size imbalance**, when either the subject or the object of the relation is way bigger at the image level than the rest of the pair. In that case, the VLM can predict a relation that only describes the bigger object, completly ignoring the relation with the smaller one.
2. VLMs struggle with **distant pairs**. When the subject and object are split far appart at the image level the VLM is lost and is not able to correclty identified the relation most of the time.
3. VLMs struggle with **inverse relations** and the passive form of verbs. In image captions, on which VLMs are trained on, the form of verbs is mostly active (i.e. person riding surfboard) which bias models to a unique direction of relation for certain subject-object pairs. For instance, given the pair (subject: surfboard ; object: person) models will still predict "riding" even though the correct form is "being ridden by".

## 3. The FG-OV SG Dataset

To help the development of Open-Vocabulary SGG, we finally present a new synthetic dataset, the Fine-Grained Open Vocabulary Scene Graphs (FG-OV SG) dataset. This dataset has been generated using the LlaVa-OneVision 7B VLM on a set of images from COCO and Objects365 datasets. To ensure high-quality and diverse relations, we randomly sample pairs using constraints that we have identified in the limitations (see section 2. above). This process resulted in a data distribution which is less long-tailed than previous with more fine-grained annotations. When used as pre-training for the Open-Vocabulary SGG model RLIPv2, our FG-OV SG dataset leads to better results than the baseline synthetic data used by previous work.

<img align="left" width="48%" width="1118" height="812" alt="image" src="https://github.com/user-attachments/assets/0732eaac-9575-4533-96bb-e3de073d6606" />
<img align="right" width="48%" width="1121" height="735" alt="image" src="https://github.com/user-attachments/assets/41a0d054-08ef-4d33-bf9e-27c9dc773702" />


