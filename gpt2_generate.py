import os
import time
import datetime
# from google.colab import drive

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
# % matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TextDataset,DataCollatorForLanguageModeling

import nltk
# nltk.download('punkt')

import pdb
from tqdm import tqdm

# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))



# model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load('model_save_50/pytorch_model.bin'))
model.eval()

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()


# prompt = "<|startoftext|>"

# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)

# print(generated)
inputs = []
outputs = []
gts = []

test_df = pd.read_csv('data/test.csv')

# pdb.set_trace()

for row in test_df.iterrows():
    text = row[1]['original_text']
    gt = row[1]['reframed_text']

    prompt = f"<|startoftext|> {text} \nreframed: "

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    sample_outputs = model.generate(
                                    generated,
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,
                                    top_k=50,
                                    max_length = 300,
                                    top_p=0.95,
                                    # num_return_sequences=3,
                                    pad_token_id=tokenizer.eos_token_id,
                                    )
                                    
    output = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)

    output = output[0].split('reframed:')[-1].strip()
    inputs.append(text)
    outputs.append(output)
    gts.append(gt)

    print('---------------------------------')
    print('input: ', text)
    print('output: ', output)
    print('gt: ', gt)

out_df = pd.DataFrame(list(zip(inputs, outputs, gts)),
              columns=['input','output', 'gt'])

out_df.to_csv('generation_res.csv')