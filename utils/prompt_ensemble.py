import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

texture_list = ['carpet', 'leather','grid',
                  'tile', 'wood']

class_mapping = {"macaroni1":"macaroni",
"macaroni2":"macaroni",
# "metal_nut":"metal nut",
#"pcb1":"pcb",#"printed circuit board",
#"pcb2":"pcb",#"printed circuit board",
#"pcb3":"pcb",#"printed circuit board",
#"pcb4":"pcb",#"printed circuit board",
#"pipe_fryum":"pipe fryum",
}

def encode_text_with_prompt_ensemble(clip_model, objs, tokenizer, device, dataset='mvtec', clip_type='CLIP'):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [state_normal, state_anomaly]
    #prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    text_prompts_list = {}
    for obj in objs:
        # print(objs)
        # prompt_templates = templates
        if obj in texture_list:
            prompt_templates = inds_temp+text_temp+surf_temp
        else:
            prompt_templates = img_temp
        text_features = []
        # text_features_list = []
        prompt_state[0] = state_normal #+ class_state_normal[obj]
        if dataset == 'mvtec':
            prompt_state[1] = state_anomaly + class_state_abnormal[obj] + class_state[obj] 
        else:
            prompt_state[1] = state_anomaly + visa_class_state_abnormal[obj] # + class_state[obj] 
        for i in range(len(prompt_state)):
            # if i == 0:
            #     if obj in texture_list:
            #         prompt_templates = inds_temp+text_temp+surf_temp
            #     else:
            #         prompt_templates = inds_temp+img_temp+mnf_temp
            # else:
            #     prompt_templates = templates
            if obj in class_mapping:
                prompted_state = [state.format(class_mapping[obj]) for state in prompt_state[i]]
            else:
                prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for template in prompt_templates:
                for s in prompted_state:
                #for template in prompt_templates:
                    prompted_sentence.append(template.format(s))

            # print("prompted_sentence: ", prompted_sentence)
            if clip_type == 'CLIP':
                prompted_sentence = tokenizer(prompted_sentence).to(device)
                class_embeddings = clip_model.encode_text(prompted_sentence)
            elif clip_type == 'SigLIP' or clip_type == 'SigLIP2':
                # prompted_sentence = tokenizer(prompted_sentence,  padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
                prompted_sentence = tokenizer(prompted_sentence,  padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
                # print("prompted_sentence: ", prompted_sentence)
                class_embeddings = clip_model.get_text_features(**prompted_sentence)

            # 5 torch.Size([154, 768]) 5 torch.Size([2772, 768]) SigLIP
            # 5 torch.Size([154, 768]) ·5 torch.Size([2772, 768]) CLIP
            # print("prompted_sentence: ", prompted_sentence)
            # print("class_embeddings:", class_embeddings)
            # print(5, class_embeddings.shape)
            class_embedding = class_embeddings.mean(dim=0)
            # print(6,class_embedding.shape)
            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
            # text_features_list.append(class_embeddings)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features
        # text_features_list = torch.stack(text_features_list, dim=2).to(device)

        # text_prompts_list[obj] = text_features_list

    return text_prompts, text_prompts_list

state_normal = [#"{}",
                #"undamaged {}",
                "normal {}",
                "flawless {}",
                "perfect {}",
                "unblemished {}",
                "{} without flaw",
                "{} without defect",
                "{} without damage",
                ]

state_anomaly = ["damaged {}",
                 #"flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

templates = ["a cropped photo of the {}",
             "a cropped photo of a {}",
             "a close-up photo of a {}",
             "a close-up photo of the {}",
             "a bright photo of a {}",
             "a bright photo of the {}",
             "a dark photo of the {}",
             "a dark photo of a {}",
             "a jpeg corrupted photo of a {}",
             "a jpeg corrupted photo of the {}",
             "a blurry photo of the {}",
             "a blurry photo of a {}",
             "a photo of a {}",
             "a photo of the {}",
             "a photo of a small {}",
             "a photo of the small {}",
             "a photo of a large {}",
             "a photo of the large {}",
             "a photo of the {} for visual inspection",
             "a photo of a {} for visual inspection",
             "a photo of the {} for anomaly detection", 
             "a photo of a {} for anomaly detection",]

inds_temp = ["a cropped industrial photo of the {}",
             "a cropped industrial photo of a {}",
             "a close-up industrial photo of a {}",
             "a close-up industrial photo of the {}",
             "a bright industrial photo of a {}",
             "a bright industrial photo of the {}",
             "a dark industrial photo of the {}",
             "a dark industrial photo of a {}",
             "a jpeg corrupted industrial photo of a {}",
             "a jpeg corrupted industrial photo of the {}",
             "a blurry industrial photo of the {}",
             "a blurry industrial photo of a {}",
             "an industrial photo of a {}",
             "an industrial photo of the {}",
             "an industrial photo of a small {}",
             "an industrial photo of the small {}",
             "an industrial photo of a large {}",
             "an industrial photo of the large {}",
             "an industrial photo of the {} for visual inspection",
             "an industrial photo of a {} for visual inspection",
             "an industrial photo of the {} for anomaly detection", 
             "an industrial photo of a {} for anomaly detection",]

img_temp = ["a cropped industrial image of the {}",
             "a cropped industrial image of a {}",
             "a close-up industrial image of a {}",
             "a close-up industrial image of the {}",
             "a bright industrial image of a {}",
             "a bright industrial image of the {}",
             "a dark industrial image of the {}",
             "a dark industrial image of a {}",
             "a jpeg corrupted industrial image of a {}",
             "a jpeg corrupted industrial image of the {}",
             "a blurry industrial image of the {}",
             "a blurry industrial image of a {}",
             "an industrial image of a {}",
             "an industrial image of the {}",
             "an industrial image of a small {}",
             "an industrial image of the small {}",
             "an industrial image of a large {}",
             "an industrial image of the large {}",
             "an industrial image of the {} for visual inspection",
             "an industrial image of a {} for visual inspection",
             "an industrial image of the {} for anomaly detection", 
             "an industrial image of a {} for anomaly detection",]

mnf_temp = ["a cropped manufacturing image of the {}",
             "a cropped manufacturing image of a {}",
             "a close-up manufacturing image of a {}",
             "a close-up manufacturing image of the {}",
             "a bright manufacturing image of a {}",
             "a bright manufacturing image of the {}",
             "a dark manufacturing image of the {}",
             "a dark manufacturing image of a {}",
             "a jpeg corrupted manufacturing image of a {}",
             "a jpeg corrupted manufacturing image of the {}",
             "a blurry manufacturing image of the {}",
             "a blurry manufacturing image of a {}",
             "a manufacturing image of a {}",
             "a manufacturing image of the {}",
             "a manufacturing image of a small {}",
             "a manufacturing image of the small {}",
             "a manufacturing image of a large {}",
             "a manufacturing image of the large {}",
             "a manufacturing image of the {} for visual inspection",
             "a manufacturing image of a {} for visual inspection",
             "a manufacturing image of the {} for anomaly detection", 
             "a manufacturing image of a {} for anomaly detection",]

text_temp = ["a cropped textural photo of the {}",
             "a cropped textural photo of a {}",
             "a close-up textural photo of a {}",
             "a close-up textural photo of the {}",
             "a bright textural photo of a {}",
             "a bright textural photo of the {}",
             "a dark textural photo of the {}",
             "a dark textural photo of a {}",
             "a jpeg corrupted textural photo of a {}",
             "a jpeg corrupted textural photo of the {}",
             "a blurry textural photo of the {}",
             "a blurry textural photo of a {}",
             "a textural photo of a {}",
             "a textural photo of the {}",
             "a textural photo of a small {}",
             "a textural photo of the small {}",
             "a textural photo of a large {}",
             "a textural photo of the large {}",
             "a textural photo of the {} for visual inspection",
             "a textural photo of a {} for visual inspection",
             "a textural photo of the {} for anomaly detection", 
             "a textural photo of a {} for anomaly detection",]

surf_temp = ["a cropped surface photo of the {}",
             "a cropped surface photo of a {}",
             "a close-up surface photo of a {}",
             "a close-up surface photo of the {}",
             "a bright surface photo of a {}",
             "a bright surface photo of the {}",
             "a dark surface photo of the {}",
             "a dark surface photo of a {}",
             "a jpeg corrupted surface photo of a {}",
             "a jpeg corrupted surface photo of the {}",
             "a blurry surface photo of the {}",
             "a blurry surface photo of a {}",
             "a surface photo of a {}",
             "a surface photo of the {}",
             "a surface photo of a small {}",
             "a surface photo of the small {}",
             "a surface photo of a large {}",
             "a surface photo of the large {}",
             "a surface photo of the {} for visual inspection",
             "a surface photo of a {} for visual inspection",
             "a surface photo of the {} for anomaly detection", 
             "a surface photo of a {} for anomaly detection",]

# surf_temp = ["a cropped surface picture of the {}",
#              "a cropped surface picture of a {}",
#              "a close-up surface picture of a {}",
#              "a close-up surface picture of the {}",
#              "a bright surface picture of a {}",
#              "a bright surface picture of the {}",
#              "a dark surface picture of the {}",
#              "a dark surface picture of a {}",
#              "a jpeg corrupted surface picture of a {}",
#              "a jpeg corrupted surface picture of the {}",
#              "a blurry surface picture of the {}",
#              "a blurry surface picture of a {}",
#              "a surface picture of a {}",
#              "a surface picture of the {}",
#              "a surface picture of a small {}",
#              "a surface picture of the small {}",
#              "a surface picture of a large {}",
#              "a surface picture of the large {}",
#              "a surface picture of the {} for visual inspection",
#              "a surface picture of a {} for visual inspection",
#              "a surface picture of the {} for anomaly detection", 
#              "a surface picture of a {} for anomaly detection",]


class_state = {'object':['{} with large breakage', '{} with small breakage', '{} with contamination',
  '{} with defect', '{} with anomaly',
  '{} with hole', 
  '{} with crack', 'cracked {}', 'cut {}', '{} with hole', '{} with print', 
  'color-varied {}', 'cut {}', 'folded {}', '{} with fold', '{} with glue', '{} with poke', 
  'bent {}', '{} with bent wire', 'missing {}', '{} with missing wire', 
  '{} with crack', 
  'broken {}',  '{} with thread',
  'combined {}', '{} with contamination', 'cracked {}', '{} with crack', '{} with faulty imprint', '{} with scratch', 
  '{} with bent lead', 'bent {}', '{} with cut lead', 'damaged {}', 
  'bent {}', 'color-varied {}', 'flipped {}', '{} with scratch', 
  '{} with manipulated front',  '{} with scratch neck', 
  '{} with broken teeth', 'combined {}',   'rough {}', '{} with split teeth', '{} with squeezed teeth', 
  'cracked {}', '{} with crack', '{} with glue strip', 'rough {}', 
 '{} with hole', '{} with liquid']
 }


class_state_abnormal = {
    'object': ['{} with large breakage', '{} with small breakage', '{} with contamination',
        '{} with defect', '{} with anomaly',
        '{} with hole', '{} with color stain', '{} with metal contamination', '{} with thread residue', '{} with thread', '{} with cut',
        '{} with crack', '{} with cut', '{} with hole', '{} with print',
        '{} with color stain', '{} with cut', '{} with fold', '{} with glue', '{} with poke',
        '{} with bent wire', '{} with missing part', '{} with missing wire', '{} with cut', '{} with poke',
        '{} with crack', '{} with faulty imprint', '{} with poke', '{} with scratch', '{} squeezed with compression',
        '{} with breakage',  '{} with thread residue', '{} with thread', '{} with metal contamination', '{} with glue', '{} with a bent shape',
        '{} with color stain', '{} with contamination', '{} with crack', '{} with faulty imprint', '{} with scratch', '{} with abnormal type',
        '{} with bent lead', '{} with cut lead', '{} with damage', '{} with misplaced transistor',
        '{} with a bent shape ', '{} with color stain', '{} with a flipped orientation', '{} with scratch',
        '{} with manipulated front',  '{} with scratch neck', '{} with scratch head',
        '{} with broken teeth', '{} with fabric border', '{} with defect fabric', '{} with broken fabric', '{} with split teeth', '{} with squeezed teeth',
        '{} with crack', '{} with glue strip', '{} with gray stroke', '{} with oil', '{} with rough surface',
        '{} with color stain', '{} with hole', '{} with scratch', '{} with liquid']}

class_state_normal = {
    'object': ['{} without large breakage', '{} without small breakage', '{} without contamination',
     '{} without defect', '{} without anomaly',
     '{} without hole', '{} without color stain', '{} without metal contamination', '{} without thread residue', '{} without thread', '{} without cut',
     '{} without crack', '{} without cut', '{} without hole', '{} without print',
     '{} without color stain', '{} without cut', '{} without fold', '{} without glue', '{} without poke',
     '{} without bent wire', '{} without missing part', '{} without missing wire', '{} without cut', '{} without poke',
     '{} without crack', '{} without faulty imprint', '{} without poke', '{} without scratch', '{} squeezed without compression',
     '{} without breakage',  '{} without thread residue', '{} without thread', '{} without metal contamination', '{} without glue', '{} without a bent shape',
     '{} without color stain', '{} without contamination', '{} without crack', '{} without faulty imprint', '{} without scratch', '{} without abnormal type',
     '{} without bent lead', '{} without cut lead', '{} without damage', '{} without misplaced transistor',
     '{} without a bent shape ', '{} without color stain', '{} without a flipped orientation', '{} without scratch',
     '{} without manipulated front',  '{} without scratch neck', '{} without scratch head',
     '{} without broken teeth', '{} without fabric border', '{} without defect fabric', '{} without broken fabric', '{} without split teeth', '{} without squeezed teeth',
     '{} without crack', '{} without glue strip', '{} without gray stroke', '{} without oil', '{} without rough surface',
     '{} without color stain', '{} without hole', '{} without scratch', '{} without liquid']}

visa_class_state_abnormal = {
    'object':[ '{} with melded wax', '{} with foreign particals', '{} with extra wax', '{} with chunk of wax missing', '{} with weird candle wick', '{} with damaged corner of packaging', '{} with different colour spot',
     '{} with scratch', '{} with discolor', '{} with misshape', '{} with leak', '{} with bubble',
     '{} with breakage', '{} with small scratches', '{} with burnt', '{} with stuck together', '{} with spot',
     '{} with corner missing', '{} with scratches', '{} with chunk of gum missing', '{} with colour spot', '{} with cracks',
     '{} with color spot', '{} with fryum stuck together', 
     '{} with small chip around edge',
     '{} with bent', '{} with missing', '{} with melt',
     '{} with extra', '{} with wrong place', '{} with damage', '{} with dirt']}