import argparse
import os
import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m') #Evaluation model name
parser.add_argument('--generation_model', type=str, default='opt-350m') #Generation model name
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--dataset', type=str, default='trivia_qa') #Dataset name
parser.add_argument('--project', type=str, default='opt_350m') #Project name
args = parser.parse_args() #Parse the arguments

device = 'cuda' #Set the device to cuda
import config

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value) #Set the PYTHONHASHSEED environment variable to the seed value
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value) #Set the random seed to the seed value
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value) #Set the numpy random seed to the seed value

#Fix torch random seed
torch.manual_seed(seed_value) #Set the torch random seed to the seed value

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache #Set the HF_DATASETS_CACHE environment variable to the hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.evaluation_model}", #Load the evaluation model
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.evaluation_model}",
                                          use_fast=False,
                                          cache_dir=config.data_dir)

wandb.init(project=args.project, id=args.run_id, config=args, resume='allow') #Initialize wandb

run_name = wandb.run.name #Get the run name

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b'] #List of opt models

with open(f'{config.output_dir}/sequences/{run_name}/{args.dataset}_{args.generation_model}_generations_small.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/semantic_similarities/{run_name}/{args.dataset}_{args.generation_model}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


def get_neg_loglikelihoods(model, sequences): #Get the negative log likelihoods

    with torch.no_grad():
        result = [] #Initialize the result list
        for sample in sequences: #For each sample in the sequences
            result_dict = {} #Initialize the result dictionary
            prompt = sample['prompt'] #Get the prompt token ids
            if 'cleaned_generations' in sample: #If the cleaned generations are in the sample
                generations = sample['cleaned_generations'].to(device) #Get the cleaned generations token ids
            else:
                generations = sample['generations'].to(device) #Get the generations token ids
            id_ = sample['id'] #Get the id

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],)) #Initialize the average negative log likelihoods to 0. The shape is (number of generations, 1)
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],)) #Initialize the average unconditioned negative log likelihoods to 0. The shape is (number of generations, 1)
            neg_log_likelihoods = torch.zeros((generations.shape[0],)) #Initialize the negative log likelihoods to 0. The shape is (number of generations, 1)
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],)) #Initialize the negative unconditioned log likelihoods to 0. The shape is (number of generations, 1)
            pointwise_mutual_information = torch.zeros((generations.shape[0],)) #Initialize the pointwise mutual information to 0. The shape is (number of generations, 1)
            sequence_embeddings = [] #Initialize the sequence embeddings list

            for generation_index in range(generations.shape[0]): #For each generation in the generations
                prompt = prompt[prompt != tokenizer.pad_token_id] #Remove the pad token ids from the prompt TODO: Check what is the token id of pad token. The prompt is appended with token 1 so they should be removed, so does that mean pad_token_id = 1?
                generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id] #Remove the pad token ids from the generation

                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone() #Clone the generation token ids
                target_ids[:len(prompt)] = -100 #Set the target ids to -100 for the prompt
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                generation_only = generation.clone()[(len(prompt) - 1):] #TODO: Check if the boundary is correct. 
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)), #Unconditioned model output
                                                   labels=generation_only,
                                                   output_hidden_states=True)
                hidden_states = model_output['hidden_states'] #Get the hidden states
                average_neg_log_likelihood = model_output['loss'] #Get the average negative log likelihood

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss'] #Get the average unconditioned negative log likelihood
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood #Set the average negative log likelihood to the average negative log likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood #Set the average unconditioned negative log likelihood to the average unconditioned negative log likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt)) #Set the negative log likelihood to the average negative log likelihood * (length of generation - length of prompt)
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt)) #Set the negative unconditioned log likelihood to the average unconditioned negative log likelihood * (length of generation - length of prompt)
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index] #Set the pointwise mutual information to the negative log likelihood + the negative unconditioned log likelihood

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1) #Get the average of the last layer token embeddings. Shape is (1, hidden_size) TODO: Check if this is correct or not.
                sequence_embeddings.append(average_of_last_layer_token_embeddings) #Append the average of the last layer token embeddings to the sequence embeddings list

            most_likely_generation = sample['most_likely_generation_ids'].to(device) #Get the most likely generation token ids
            target_ids = most_likely_generation.clone() #Clone the most likely generation token ids
            target_ids[:len(prompt)] = -100 #Set the target ids to -100 for the prompt
            model_output = model(torch.reshape(most_likely_generation, (1, -1)), #Model output
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states'] #Get the hidden states
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss'] #Get the average negative log likelihood of the most likely generation
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1) #Get the most likely generation embedding. Shape is (1, hidden_size) TODO: Check if this is correct or not.

            second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device) #Get the second most likely generation token ids
            target_ids = second_most_likely_generation.clone() #Clone the second most likely generation token ids
            target_ids[:len(prompt)] = -100 #Set the target ids to -100 for the prompt
            model_output = model(torch.reshape(second_most_likely_generation, (1, -1)), #Model output
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states'] #Get the hidden states
            average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss'] #Get the average negative log likelihood of the second most likely generation
            second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1) #Get the second most likely generation embedding. Shape is (1, hidden_size) TODO: Check if this is correct or not.

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt)) #Set the negative log likelihood of the most likely generation to the average negative log likelihood of the most likely generation * (length of most likely generation - length of prompt)

            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt #Set the prompt to the prompt
            result_dict['generations'] = generations #Set the generations to the generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods #Set the average negative log likelihoods to the average negative log likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods #Set the negative log likelihoods to the negative log likelihoods
            result_dict['sequence_embeddings'] = most_likely_generation_embedding #Set the sequence embeddings to the most likely generation embedding
            result_dict['most_likely_sequence_embedding'] = most_likely_generation #Set the most likely sequence embedding to the most likely generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods #Set the average unconditioned negative log likelihoods to the average unconditioned negative log likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods #Set the negative unconditioned log likelihoods to the negative unconditioned log likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information #Set the pointwise mutual information to the pointwise mutual information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen #Set the average negative log likelihood of the most likely generation to the average negative log likelihood of the most likely generation
            result_dict[
                'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen #Set the average negative log likelihood of the second most likely generation to the average negative log likelihood of the second most likely generation
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen #Set the negative log likelihood of the most likely generation to the negative log likelihood of the most likely generation
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device) #Set the semantic set ids to the semantic set ids
            result_dict['id'] = id_ #Set the id to the id
            result.append(result_dict) #Append the result dictionary to the result list

        return result #Return the result list


likelihoods = get_neg_loglikelihoods(model, sequences) #Get the negative log likelihoods

with open(f'{config.output_dir}/likelihoods/{run_name}/{args.dataset}_{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl',
          'wb') as outfile: #Open the output file for writing
    pickle.dump(likelihoods, outfile) #Dump the likelihoods to the output file

