# Read generation results
import argparse
import os
import pickle
import random

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
#sns.color_palette("pastel")
import wandb
from config import device_map
from dotenv import load_dotenv
load_dotenv()

# Set a seed value
seed_value = 10 # Set the seed value to 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value) # Set the PYTHONHASHSEED environment variable to the seed value
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value) # Set the random seed to the seed value
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value) # Set the numpy random seed to the seed value

device = torch.device('cuda') # Set the device to cuda

#Fix torch random seed
torch.manual_seed(seed_value) # Set the torch random seed to the seed value

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache # Set the HF_DATASETS_CACHE environment variable to the hf_datasets_cache

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-1.3b') #Generation model name
parser.add_argument('--run_id_for_few_shot_prompt', type=str, default='run_1')
parser.add_argument('--run_id_for_evaluation', type=str, default='run_1')
parser.add_argument('--project', type=str, default='opt_350m') #Project name
parser.add_argument('--model', type=str, default='opt-350m') #Model name
parser.add_argument('--dataset', type=str, default='trivia_qa') #Dataset name
args = parser.parse_args()

wandb.init(project=args.project, id=args.run_id_for_few_shot_prompt, config=args, resume='allow')
model_name = args.model #Get the model name

generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generation_model}", use_fast=False, cache_dir=config.data_dir) #Load the generation tokenizer
model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.generation_model}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda() #Load the model TODO: Which model to load here?

# #region agent log
import os as _os
_debug_log_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".cursor", "debug.log")
_os.makedirs(_os.path.dirname(_debug_log_path), exist_ok=True)
import json as _json
def _debug_log(loc, msg, data, hyp): open(_debug_log_path, 'a').write(_json.dumps({"location": loc, "message": msg, "data": data, "hypothesisId": hyp, "sessionId": "debug-session"}) + '\n')
_debug_log("get_prompting_based_uncertainty.py:model_load", "Model and tokenizer loaded", {"vocab_size": model.config.vocab_size, "max_position_embeddings": model.config.max_position_embeddings, "tokenizer_vocab_size": len(generation_tokenizer)}, "A,C")
# #endregion

if args.generation_model == 'opt-30b':
    accelerate.dispatch_model(model, device_map=device_map) #Dispatch the model to the device map
    print(model.hf_device_map) #Print the device map
    device = torch.device('cuda:1') #Set the device to cuda:1

run_name = wandb.run.name #Get the run name

with open(f'{config.output_dir}/sequences/{run_name}/{args.dataset}_{model_name}_generations.pkl', 'rb') as infile:
    sequences_for_few_shot_prompt = pickle.load(infile)

wandb.finish() #Finish the wandb run

# Build few shot prompt

subset_of_sequences_for_few_shot_prompt = sequences_for_few_shot_prompt[-10:] #Get the last 10 sequences for the few shot prompt
number_of_few_shot_samples = 5 #Get the number of few shot samples

prompt_template = 'Question: {} \n Here are some ideas that were brainstormed:{}\n Possible answer:{}\n Is the possible answer:\n (A) True\n (B) False\n The possible answer is:' #Prompt template
few_shot_promopt = '' #Few shot prompt
for sequence in subset_of_sequences_for_few_shot_prompt:
    question = sequence['question'] #Get the question (text)
    question = question.split('Question: ')[-1].split('Answer: ')[0] #Get the question
    prompt = sequence['prompt'] #Get the prompt (token ids)
    generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples]) #Get the generated texts

    most_likely_answer = sequence['most_likely_generation'] #Get the most likely answer
    correct = ' True' if sequence['rougeL_to_target'] > 0.3 else ' False' # True if the rougeL_to_target is greater than 0.3, False otherwise
    few_shot_promopt += prompt_template.format(question, generated_texts, most_likely_answer) + correct + '\n' #Add the prompt to the few shot prompt

# Build prompt for question
labels_across_datasets = [] #List to store the labels across the datasets
p_trues_across_datasets = [] #List to store the p_trues across the datasets

n_samples_to_use = 2000

with torch.no_grad():

    aurocs = [] #List to store the aurocs
    p_trues = [] #List to store the p_trues
    corrects = [] #List to store the corrects
    for sequence in tqdm(sequences_for_few_shot_prompt[:n_samples_to_use]):

        question = sequence['question'] #Get the question (text) 
        if 'Question: ' in question:
            question = question.split('Question: ')[-1].split('Answer: ')[0] #Get the question
        else:
            question = question.split('Q: ')[-1].split('A: ')[0]

        generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples]) #Get the generated texts
        most_likely_answer = sequence['most_likely_generation'] #Get the most likely answer
        correct = 1.0 if sequence['rougeL_to_target'] > 0.3 else 0.0 # 1.0 if the rougeL_to_target is greater than 0.3, 0.0 otherwise
        base_prompt = prompt_template.format(question, generated_texts, most_likely_answer) #Get the base prompt
        prompt_true = few_shot_promopt + prompt_template.format(question, generated_texts, most_likely_answer) + ' True' #Get the prompt true

        # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
        tokenized_base_prompt = generation_tokenizer(base_prompt)['input_ids'] #Get the tokenized base prompt
        tokenized_prompt_true = torch.tensor(generation_tokenizer(prompt_true)['input_ids'], device=device)

        target_ids_true = tokenized_prompt_true.clone() #Clone the tokenized prompt true
        target_ids_true[:len(tokenized_base_prompt)] = -100 #Set the target ids to -100 for the base prompt. TODO: This is wrong right? I think we should set the target ids to -100 for the few shot prompt + base prompt.

        # #region agent log fix: Truncate sequence if it exceeds model's max position embeddings
        _max_len = model.config.max_position_embeddings
        if len(tokenized_prompt_true) > _max_len:
            _truncate_amount = len(tokenized_prompt_true) - _max_len
            tokenized_prompt_true = tokenized_prompt_true[_truncate_amount:]  # Keep the end (includes " True")
            target_ids_true = target_ids_true[_truncate_amount:]  # Truncate labels correspondingly
            _debug_log(f"get_prompting_based_uncertainty.py:truncation", "Sequence truncated", {"original_len": _truncate_amount + _max_len, "new_len": _max_len, "truncate_amount": _truncate_amount}, "C")
        # #endregion

        # #region agent log
        _iter_idx = len(p_trues)
        _max_token_id = int(tokenized_prompt_true.max().item())
        _min_token_id = int(tokenized_prompt_true.min().item())
        _seq_len = len(tokenized_prompt_true)
        _labels_shape = target_ids_true.shape
        _input_shape = torch.reshape(tokenized_prompt_true, (1, -1)).shape
        _non_masked_labels = target_ids_true[target_ids_true != -100]
        _max_label_id = int(_non_masked_labels.max().item()) if len(_non_masked_labels) > 0 else -1
        _min_label_id = int(_non_masked_labels.min().item()) if len(_non_masked_labels) > 0 else -1
        _debug_log(f"get_prompting_based_uncertainty.py:iter_{_iter_idx}", "Pre-model call tensors", {"iter": _iter_idx, "seq_len": _seq_len, "max_token_id": _max_token_id, "min_token_id": _min_token_id, "max_label_id": _max_label_id, "min_label_id": _min_label_id, "labels_shape": str(_labels_shape), "input_shape": str(_input_shape), "vocab_size": model.config.vocab_size, "max_pos_embed": model.config.max_position_embeddings}, "A,B,C,D,E")
        # #endregion

        # #region agent log
        try:
            model_output_true = model(torch.reshape(tokenized_prompt_true, (1, -1)), labels=target_ids_true) #Get the model output 
            loss_true = model_output_true.loss #Get the loss
            _debug_log(f"get_prompting_based_uncertainty.py:iter_{_iter_idx}_success", "Model call succeeded", {"iter": _iter_idx, "loss": float(loss_true.item())}, "ALL")
        except Exception as _e:
            _debug_log(f"get_prompting_based_uncertainty.py:iter_{_iter_idx}_error", "Model call FAILED", {"iter": _iter_idx, "error": str(_e), "seq_len": _seq_len, "max_token_id": _max_token_id, "min_token_id": _min_token_id, "vocab_size": model.config.vocab_size, "max_pos_embed": model.config.max_position_embeddings, "labels_shape": str(_labels_shape)}, "A,B,C,D,E")
            raise
        # #endregion

        p_trues.append(loss_true.item()) #Append the loss to the p_trues list
        corrects.append(correct) #Append the correct to the corrects list

        labels_across_datasets += corrects #Append the corrects to the labels across the datasets
        p_trues_across_datasets += p_trues #Append the p_trues to the p_trues across the datasets

    p_true_auroc = sklearn.metrics.roc_auc_score(1 - torch.tensor(corrects), torch.tensor(p_trues)) #Get the p_true auroc
    print(f"P_true auroc: {p_true_auroc}") #Print the p_true auroc
    # Store p_true aurocs in a pickle file
    with open(f'{config.output_dir}/prompting_based_uncertainty/{run_name}/{args.dataset}_{model_name}_p_true_aurocs.pkl', 'wb') as outfile:
        pickle.dump(p_true_auroc, outfile) #Dump the p_true auroc to the output file

