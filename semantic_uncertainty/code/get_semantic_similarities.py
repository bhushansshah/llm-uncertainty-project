import argparse
import csv
import os
import pickle
import random

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import wandb
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m') #Generation model
parser.add_argument('--run_id', type=str, default='run_1') #Run id
parser.add_argument("--project", type=str, default='opt_350m') #Project name
parser.add_argument('--dataset', type=str, default='trivia_qa') #Dataset name
args = parser.parse_args()

device = 'cuda' #Set the device to GPU

# Set a seed value
seed_value = 10 #Set the seed value
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value) #Set the PYTHONHASHSEED environment variable to the seed value
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value) #Set the numpy random seed to the seed value

#Fix torch random seed
torch.manual_seed(seed_value) #Set the torch random seed to the seed value

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache #Set the HF_DATASETS_CACHE environment variable to the cache directory

generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.generation_model}", use_fast=False, cache_dir=config.data_dir) #Load the generation tokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli") # Load the tokenizer for semantic similarity evaluation
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda() # Load the model for semantic similarity evaluation

wandb.init(project=args.project, id=args.run_id, config=args, resume='allow') #Initialize wandb

run_name = wandb.run.name #Get the run name

with open(f'{config.output_dir}/sequences/{run_name}/{args.dataset}_{args.generation_model}_generations.pkl', 'rb') as infile: #Load the generations
    sequences = pickle.load(infile)

result_dict = {} #Initialize the result dictionary

meteor = evaluate.load('meteor') #Load the meteor metric

deberta_predictions = [] # Initialize the deberta predictions list

for sample in tqdm(sequences): #For each sample in the sequences
    question = sample['question'] #Get the question
    if 'cleaned_generated_texts' in sample: #If the cleaned generated texts are in the sample
        generated_texts = sample['cleaned_generated_texts'] #Get the cleaned generated texts
    else: #If the cleaned generated texts are not in the sample
        generated_texts = sample['generated_texts'] #Get the generated texts

    id_ = sample['id'][0] #Get the id

    unique_generated_texts = list(set(generated_texts)) #Get the unique generated texts

    answer_list_1 = [] # Initialize the answer list 1
    answer_list_2 = [] # Initialize the answer list 2
    has_semantically_different_answers = False 
    inputs = [] # Initialize the inputs list
    syntactic_similarities = {} # Initialize the syntactic similarities dictionary
    rouge_types = ['rouge1', 'rouge2', 'rougeL'] # List of rouge types
    for rouge_type in rouge_types:
        syntactic_similarities[rouge_type] = 0.0 # Initialize the syntactic similarities to 0.0

    semantic_set_ids = {} # Initialize the semantic set ids dictionary
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index # Set the semantic set ids to the index

    print('Number of unique answers:', len(unique_generated_texts))

    if len(unique_generated_texts) > 1: #If there are more than one unique generated text

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts): #For each reference answer
            for j in range(i + 1, len(unique_generated_texts)): #For each other reference answer

                answer_list_1.append(unique_generated_texts[i]) #Append the first reference answer to the answer list 1
                answer_list_2.append(unique_generated_texts[j]) #Append the second reference answer to the answer list 2

                qa_1 = question + ' ' + unique_generated_texts[i] #Form the first question and answer
                qa_2 = question + ' ' + unique_generated_texts[j] #Form the second question and answer

                input = qa_1 + ' [SEP] ' + qa_2 #Form the input to the semantic similarity evaluation
                inputs.append(input) #Append the input to the inputs list
                encoded_input = tokenizer.encode(input, padding=True) #Encode the input
                prediction = model(torch.tensor([encoded_input]).to("cuda"))['logits'] #Get the prediction - this is a tensor of shape (1, 3) which contains the logits for the three classes: 0 (contradiction), 1 (neutral), 2 (entailment)
                predicted_label = torch.argmax(prediction, dim=1) #Get the predicted label

                reverse_input = qa_2 + ' [SEP] ' + qa_1 #Form the reverse input to the semantic similarity evaluation
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True) #Encode the reverse input
                reverse_prediction = model(torch.tensor([encoded_reverse_input]).to('cuda'))['logits'] #Get the reverse prediction 
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1) #Get the reverse predicted label

                deberta_prediction = 1 # Initialize the deberta prediction to 1
                print(qa_1, qa_2, predicted_label, reverse_predicted_label) 
                if 0 in predicted_label or 0 in reverse_predicted_label: # If the predicted label is 0 or the reverse predicted label is 0, then the answers are semantically different
                    has_semantically_different_answers = True # Set the has semantically different answers to True
                    deberta_prediction = 0 # Set the deberta prediction to 0

                else: # If the predicted label is not 0 or the reverse predicted label is not 0, then the answers are semantically the same
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]] # Set the semantic set ids to the index of the first reference answer TODO: Check if this is correct or not. What is the meaning of deberta_prediction = 1? Why is nuetral prediction considered as semantically the same?

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction]) # Append the deberta prediction to the deberta predictions list

        rouge = evaluate.load('rouge') # Load the rouge metrics

        # Evalauate syntactic similarity
        answer_list_1 = [] # Initialize the answer list 1
        answer_list_2 = [] # Initialize the answer list 2
        for i in generated_texts: #For each generated text
            for j in generated_texts: #For each other generated text
                if i != j: #If the generated text is not the same as the other generated text
                    answer_list_1.append(i) #Append the generated text to the answer list 1
                    answer_list_2.append(j) #Append the other generated text to the answer list 2

        results = rouge.compute(predictions=answer_list_1, references=answer_list_2) #Compute the rouge scores

        for rouge_type in rouge_types: 
            syntactic_similarities[rouge_type] = results[rouge_type] #Set the syntactic similarities to the rouge scores

    result_dict[id_] = {
        'syntactic_similarities': syntactic_similarities,
        'has_semantically_different_answers': has_semantically_different_answers
    }
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids

with open(f'{config.output_dir}/semantic_similarities/{run_name}/{args.dataset}_{args.generation_model}_deberta_predictions.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['qa_1', 'qa_2', 'prediction'])
    writer.writerows(deberta_predictions)

print(result_dict)

with open(f'{config.output_dir}/semantic_similarities/{run_name}/{args.dataset}_{args.generation_model}_generations_similarities.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)

