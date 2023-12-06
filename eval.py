import argparse
from nlgeval import compute_individual_metrics
import pandas as pd
# from rouge_score import rouge_scorer
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
#from bert_score import score
import os
from nlgeval import NLGEval
import pdb
# from cal_eval import *
# from perspective import PerspectiveAPI
import time
# from cal_novelty import *
import torch
from tqdm import tqdm
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, GPT2Tokenizer
from datasets import load_dataset, load_metric

import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--kpq', type=int, default=100)
parser.add_argument('--file', type=str, default='')
parser.add_argument('--save_json', type=bool, default='False')
args = parser.parse_args()

print('--------------------------')
# scorer = rouge_scorer.RougeScorer('rouge2', use_stemmer=True)
print(args.file)
path = os.path.dirname(args.file)

file, file_ext = (os.path.splitext(args.file))

# file = file.split('/')
# file_name = file[-2]

if file_ext == '.xlsx':
    df = pd.read_excel(args.file, engine='openpyxl')
elif file_ext == '.csv':
    df = pd.read_csv(args.file)
else:
    print('not supported')

# gt = pd.read_csv('/mnt/nas/jieun/conan/conan_test.csv')

# metric = load_metric("rouge")
# metric2 = load_metric("sacrebleu")

rouge = Rouge()

# if len(df['output']) != len(df['gt']) :
#     gt = gt.iloc[0:500]

## BLUE ###
# nlgeval = NLGEval()  # loads the models
# metrics_dict = nlgeval.compute_metrics([df['gt']], df['output'])
#######################

### SACREBLEU ###
sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=df['output'], references=df['gt'])

# sacrebleu = metric2.compute(predictions=df['output'], references=[df['gt']])
# print(sacrebleu['score'])

### ROUGE ###
scores = rouge.get_scores(df['output'], df['gt'], avg=True)



# print(f"path : {args.file}")
# print("Toxicity score: ", df['TOXICITY'].mean())
# print('bleu-2:', metrics_dict['Bleu_2'] * 100)
print('sacrebleu:', results['score'])
print('rouge-1:', scores['rouge-1']['f'] * 100)
print('rouge-2:', scores['rouge-2']['f'] * 100)
print('rouge-l:', scores['rouge-l']['f'] * 100)


if args.save_json:
    data_dict = {
        # 'bleu-2': metrics_dict['Bleu_2'] * 100,
        'sacrebleu': results['score'],
        'rouge-1': scores['rouge-1']['f'] * 100,
        'rouge-2': scores['rouge-2']['f'] * 100,
        'rouge-l': scores['rouge-l']['f'] * 100,
       
    }
    
    js = json.dumps(data_dict, indent=4)
    with open("eval.json", "w") as outfile:
        outfile.write(js)

# BERTSCORE
# This metric uses GPU
# from evaluate import load
# bertscore = load("bertscore")
# results = bertscore.compute(predictions=df['CN'], references=gt['COUNTER_NARRATIVE'], lang="en")
# print(results)
# P, R, F1 = score(df['pred_CN'], df['CN'], lang='en', verbose=True)
# print(f"System level F1 score: {F1.mean():.3f}")
