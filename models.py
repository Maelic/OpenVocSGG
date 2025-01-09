import torch

import torch
from PIL import Image
import openai
import base64

# import transformers.generation

# from models_utils import CustomGenerationMixin

# transformers.generation.GenerationMixin = CustomGenerationMixin

from transformers import AutoModelForCausalLM, AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration,PaliGemmaProcessor

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image


from vllm import LLM, SamplingParams

class BaseModel():
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.max_tokens = 300

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = None
        self.processor = None

    def get_prompt_template(self, sub, obj):
        prompt_template = "You are a robot that only outputs a relation of the form <subject, relation, object> related to an image. You reply in the following format: <sub> </sub> <rel> </rel> <obj> </obj>. You will be given the tuple <sub>, <obj> and you have to complete with the predicate <rel> </rel>. Example: <sub>person</sub> <rel>looking at</rel> <obj>dog</obj>. Now what is the relation <rel></rel> between <sub>"+sub+"</sub> and <obj>"+obj+"</obj> in this image?"
        prompt_template = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Avoid using vague predicates like 'next to.' You output the relationship in the format: <sub></sub> <rel></rel> <obj></obj>. Given a subject (<sub>) and an object (<obj>), your task is to infer and complete the predicate (<rel>) based on the image. \n Example: \n Input: <sub>person</sub> <obj>dog</obj> \n Output: <sub>person</sub> <rel>looking at</rel> <obj>dog</obj>. \n Now, determine <rel></rel> for the pair: <sub>"+sub+"</sub> and <obj>"+obj+"</obj> in this image."

        return prompt_template

    def generate(self, pair, image, prompt_template=None):
        return None

class GPT4Model(BaseModel):
    def __init__(self, model_id="gpt-4o-mini", device=None):
        super().__init__(device)

        self.client = openai.OpenAI()
        self.model_id = model_id

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def get_request(self, pair, image, img_id, prompt_template=None):
        image = image.convert("RGB")
        image.save("temp.jpg")
        with open("temp.jpg", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        user_text = "Now, determine <rel></rel> for the pair: <sub>"+pair[0]+"</sub> and <obj>"+pair[1]+"</obj> in this image."

        system_text = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Avoid using vague predicates like 'next to.' You output the relationship in the format: <sub></sub> <rel></rel> <obj></obj>. Given a subject (<sub>) and an object (<obj>), your task is to infer and complete the predicate (<rel>) based on the image. \n Example: \n Input: <sub>person</sub> <obj>dog</obj> \n Output: <sub>person</sub> <rel>looking at</rel> <obj>dog</obj>. \n "

        custom_id = self.model_id+"_"+img_id+"_"+pair[0]+"_"+pair[1]

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
        }

    def generate(self, pair, image, prompt_template=None):
        # image is in PIL format, we need to convert to base64
        image = image.convert("RGB")
        image.save("temp.jpg")
        with open("temp.jpg", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        user_text = "Now, determine <rel></rel> for the pair: <sub>"+pair[0]+"</sub> and <obj>"+pair[1]+"</obj> in this image."

        system_text1 = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Follow these refined steps:\n\n1. **Entity Identification**  \n   - Clearly identify each entity's visual characteristics within the image. Note attributes such as shape, size, and position, alongside any distinguishing features.\n\n2. **Spatial Context**  \n   - Analyze the positioning of the entities relative to each other. Consider aspects like distance, orientation, and whether entities overlap or are close in proximity.\n\n3. **Functional Context**  \n   - Assess any potential functional interactions. Identify actions or roles that may indicate how Entity 1 impacts or interacts with Entity 2.\n\n4. **Integrative Reasoning**  \n   - Determine whether the relationship is predominantly spatial or functional. Use the analysis from steps 1-3 to support your reasoning and articulate it in a concise sentence.\n\nFinally, summarize the visual relationship in the format: `<sub>Entity 1</sub> <rel>relationship</rel> <obj>Entity 2</obj>`. Example: `<sub>1_person</sub> <rel>holding</rel> <obj>2_phone</obj>`.\n\n# Output Format\n\n- Provide your response as a structured sentence summarizing the relationship, followed by the formatted statement. \n\n# Notes\n\n- Ensure that the reasoning provided is comprehensive and ties together observations from all steps.\n- Consider both tangible interactions and abstract spatial nuances when formulating the result."

        system_text = "Analyze and describe the directed visual relationship between two entities in an image, focusing on Entity 1 as the subject and Entity 2 as the object. Entity 1 is represented by the blue bounding box, while Entity 2 is the red bounding box. Avoid using vague predicates like 'next to.' You output the relationship in the format: <sub></sub> <rel></rel> <obj></obj>. Given a subject (<sub>) and an object (<obj>), your task is to infer and complete the predicate (<rel>) based on the image. \n Example: \n Input: <sub>person</sub> <obj>dog</obj> \n Output: <sub>person</sub> <rel>looking at</rel> <obj>dog</obj>. \n "

        completion = self.client.chat.completions.create(
        model=self.model_id,
        messages=[
            {
            "role": "system",
            "content": [
                    {
                        "text": system_text,
                        "type": "text"
                    }
                ]
            },
            {
            "role": "user",
            "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}", "detail": "low"}
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    }
                ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=0.5,
        max_completion_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        output_raw = completion.choices[0].message.content

        try:
            output = output_raw.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None

        return output, output_raw

class LLaVAModel(BaseModel):
    def __init__(self, device=None, region_guidance=False):
        super().__init__("llava-hf/llava-onevision-qwen2-7b-ov-hf", device) # LLaVA - OneVision model

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=self.quantization_config,
            device_map=self.device
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.region_guidance = region_guidance
    
    def predict(self, image, messages):

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )

        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,             
            pad_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_tokens
        )
        return output

    def generate(self, pair, image, prompt_template=None, image_negative=None):
        if self.region_guidance:
            assert image_negative is not None, "Region guidance requires a negative image"

        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template}
            ]}
        ]

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )

        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)
        
        model_kwargs = inputs

        if self.region_guidance:

            inputs = self.processor(
                image_negative,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)

            position_ids, attention_mask, inputs_embeds, image_sizes = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_sizes"]

            model_kwargs.update({"input_ids_blackout":position_ids,"attention_mask_blackout":attention_mask, "pixel_values_blackout": inputs_embeds, "image_sizes_blackout": image_sizes})

        outputs = self.model.generate(
            **model_kwargs,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=self.max_tokens
        )

        output = self.processor.decode(outputs[0], skip_special_tokens=True)
        out_raw = output
        # split on the end of the prompt with the assistant token
        output = output.split("assistant")[1]

        # extract only the word between <pred> </pred>
        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, out_raw

class Phi3Model(BaseModel):
    def __init__(self, device=None):
        super().__init__("microsoft/Phi-3.5-vision-instruct", device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map=self.device, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            _attn_implementation='flash_attention_2'    
        )

        self.processor =  AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            num_crops=16
        )
    
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])
        placeholder = f"<|image_1|>\n"

        messages = [
            {"role": "user", "content": placeholder+prompt_template},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device) 

        generation_args = { 
            "max_new_tokens": self.max_tokens, 
            "temperature": 0.5, 
            "do_sample": False,
        } 

        generate_ids = self.model.generate(**inputs, 
            eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res_raw = response
        # extract only the word between <pred> </pred>
        try:
            output = response.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, res_raw

class LlamaModel(BaseModel):
    def __init__(self, device=None):
        super().__init__("neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic", device)
        # model_id = "/home/maelic/Documents/OpenVocSGG/llama3_2_hf"

        self.model = LLM(model=self.model_id, max_num_seqs=1, enforce_eager=True)
    
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])

        prompt = f"<|image|><|begin_of_text|>{prompt_template}"
        sampling_params = SamplingParams(max_tokens=300)

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }
        outputs = self.model.generate(inputs, sampling_params=sampling_params)

        output = outputs[0].outputs[0].text
        res_raw = output
        # extract only the word between <pred> </pred>
        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, res_raw
    
class LlamaModel2(BaseModel):
    def __init__(self, device=None):
        super().__init__("/home/maelic/Documents/OpenVocSGG/llama3_2_hf", device)
        # model_id = "/home/maelic/Documents/OpenVocSGG/llama3_2_hf"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=self.quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template}
            ]}
        ]

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        output = self.processor.decode(output[0][inputs["input_ids"].shape[-1]:])
        # remove everything after the end token <|eot_id|>
        output = output.split("<|eot_id|>")[0]
        res_raw = output
        # extract only the word between <pred> </pred>
        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, res_raw
    
class Qwen2VLModel(BaseModel):
    def __init__(self, device=None):
        super().__init__("Qwen/Qwen2-VL-7B-Instruct", device)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=self.quantization_config,
            device_map=self.device
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_template}
            ]}
        ]

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )

        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
             padding=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,             
            max_new_tokens=self.max_tokens
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        out_raw = output
        # split on the end of the prompt with the assistant token
        # output = output.split("assistant")[1]

        # extract only the word between <pred> </pred>
        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, out_raw
    
class PaliGemmaModel(BaseModel):
    def __init__(self, device=None):
        super().__init__("google/paligemma2-3b-pt-448", device)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, 
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
        )

        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_id, 
        )
    
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])
        placeholder = f"<image> answer en "

        # messages = [
        #     {"role": "user", "content": placeholder+prompt_template},
        # ]

        # prompt = self.processor.tokenizer.apply_chat_template(
        #     messages, 
        #     tokenize=False, 
        #     add_generation_prompt=True
        # )

        prompt = placeholder+prompt_template

        model_inputs = self.processor(prompt, image, return_tensors="pt").to(torch.bfloat16).to(self.device) 

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=300, do_sample=False)
            generation = generation[0][input_len:]
            output = self.processor.decode(generation, skip_special_tokens=True)

        out_raw = output
        # split on the end of the prompt with the assistant token
        # output = output.split("assistant")[1]

        # extract only the word between <pred> </pred>
        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, out_raw
    
class InternVLModel(BaseModel):
    def __init__(self, device=None):
        super().__init__("OpenGVLab/InternVL2_5-8B-MPO", device)

        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
        
    def generate(self, pair, image, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.get_prompt_template(pair[0], pair[1])
        
        response = self.pipe((prompt_template, image))
        output = response.text
        out_raw = output

        try:
            output = output.split("<rel>")[1].split("</rel>")[0]
        except:
            output = None
        return output, out_raw