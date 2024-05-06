#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


# Load model directly
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import pandas as pd
import numpy as np
import re
import time
import warnings

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)


# In[3]:


cache_dir = "/scratch1/nnachimu/base_models"
model = AutoModelForCausalLM.from_pretrained("sinking8/finetuned_llama2", cache_dir=cache_dir, device_map=device, do_sample=True, temperature="0.2")
tokenizer = AutoTokenizer.from_pretrained("sinking8/finetuned_llama2", device=device, cache_dir=cache_dir, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# In[4]:


# del model
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.reset_max_memory_cached()
# torch.cuda.empty_cache()
# allocated_memory_bytes = torch.cuda.memory_allocated()
# allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
# print("Allocated memory:", allocated_memory_gb, "GB")


# In[5]:


def generate_text(inputs):
    prompts = []
    max_length = 0
    for text, true_label in zip(inputs['text'].values, inputs['label'].values):
        prompt = """Is the given text AI generated? Answer with Ans=1(AI generated) or Ans=0(Not AI generated): 
        %s
        Answer?
            """ % text
        prompts.append(prompt)
        max_length = max(len(tokenizer.tokenize(prompt)), max_length)
    encoded_prompts = tokenizer.batch_encode_plus(prompts, return_tensors="pt",padding=True, max_length=500).to(device)
    input_ids = encoded_prompts["input_ids"]
    attention_masks = encoded_prompts["attention_mask"]
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_masks, num_return_sequences=1, max_new_tokens=50)
    # generated_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    # print(generated_texts)
    # pattern = r"Is the text AI generated\? \(True/False\)[\s\S]*?(True|False)"
    pattern = r"Answer\?[\s\S]*?(0|1)"
    
    results = []
    for output_id in output_ids:
        generated_text = tokenizer.decode(output_id, skip_special_tokens=True)
        matches = re.findall(pattern, generated_text)
        if matches:
            results.append(1 if matches[0].strip()=="1" else 0)
        else:
            results.append(None)
    torch.cuda.empty_cache()
    return results


# In[6]:


def split_dataframe_into_batches(df, batch_size=25):
    batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batches.append(batch)
    return batches


# In[7]:


input_df = pd.read_csv('/project/vsharan_1298/CSCI-567-Project-AI-Prediction/test.csv')
input_df = input_df.sample(n=30000, random_state=42)


# In[ ]:


batches = split_dataframe_into_batches(input_df, 3)
predicted_results = []
i = 0
for batch in batches:
    start_time = time.time()
    result = generate_text(batch)
    end_time = time.time()
    time_taken = end_time - start_time
    warnings.warn("batch: {}/{} Time taken: {} seconds".format(i,len(batches), time_taken))
    i += 1
    predicted_results.extend(result)
output_file_name = "finetuned_test_prediction_llama2_reduced"
df = pd.DataFrame({
    'Text': input_df['text'].values,
    'True_Labels': input_df['label'].values,
    'Predicted_Labels': predicted_results
    })

df_filtered = df[df['Predicted_Labels'].notna()]
file_name = output_file_name
df_filtered.to_csv(file_name + ".csv", index=False)

precision, recall, f1_score, _ = precision_recall_fscore_support(df_filtered['True_Labels'].values, df_filtered['Predicted_Labels'].values, average='binary')

with open(file_name + ".txt", "w") as file:
    file.write("Precision: {} \n".format(precision))
    file.write("Recall: {}\n".format(recall))
    file.write("F1 Score: {}\n".format(f1_score))
torch.cuda.empty_cache()


# In[ ]:


torch.cuda.empty_cache()

