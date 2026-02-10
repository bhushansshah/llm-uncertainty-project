import argparse
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config
import wandb
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--project', type=str, default='opt_350m')
parser.add_argument('--dataset', type=str, default='trivia_qa')
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache #Set the cache directory for the datasets

generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generation_model}", use_fast=False, cache_dir=config.data_dir) #Load the generation tokenizer

wandb.init(project=args.project, id=args.run_id, config=args, resume='allow') #Initialize wandb

run_name = wandb.run.name

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generation_model}", use_fast=False, cache_dir=config.data_dir) #Load the tokenizer

with open(f'{config.output_dir}/sequences/{run_name}/{args.dataset}_{args.generation_model}_generations.pkl', 'rb') as infile: #Load the generations
    sequences = pickle.load(infile)

cleaned_sequences = [] #Initialize the cleaned sequences

for sample in tqdm(sequences):
    cleaned_generations = torch.ones_like(sample['generations']) #Initialize the cleaned generations to same shape as the generations (this contains the token ids of the entire prompt and the completion)
    question = sample['question'] #Get the question (text)
    generated_texts = sample['generated_texts'] #Get the generated texts (only contains the text of the completion)
    cleaned_generated_texts = [] #Initialize the cleaned generated texts

    max_len_of_generations = cleaned_generations.shape[-1] #Get the maximum length of the generations

    strings_to_filter_on = [ #List of strings to filter on
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]

    for i, generated_text in enumerate(generated_texts): #For each generated text
        for string in strings_to_filter_on: #For each string to filter on
            if string in generated_text:
                generated_text = generated_text.split(string)[0] #Split the generated text at the string and take the part before the string
        cleaned_generated_texts.append(generated_text) #Append the cleaned generated text to the list
        clean_ids = torch.cat( #Concatenate the prompt and the generated text. We remove the first token of the generated text as it is the </s>token.
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)]) #Tokenize the generated text and concatenate with the prompt
        cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations] #Set the cleaned generations to the cleaned ids

    sample['cleaned_generated_texts'] = cleaned_generated_texts #Set the cleaned generated texts to the list
    sample['cleaned_generations'] = cleaned_generations #Set the cleaned generations to the tensor
    cleaned_sequences.append(sample) #Append the cleaned sample to the list

with open(f'{config.output_dir}/sequences/{run_name}/{args.dataset}_{args.generation_model}_generations.pkl', 'wb') as outfile: #Save the cleaned sequences
    pickle.dump(cleaned_sequences, outfile) #Save the cleaned sequences
print("The cleaned sequences are saved in the pickle file")
