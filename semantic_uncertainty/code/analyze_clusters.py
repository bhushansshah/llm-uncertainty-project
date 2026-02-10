# parse arguments
import argparse
import pickle

import config
import numpy as np
import pandas as pd
import torch
from evaluate import load

parser = argparse.ArgumentParser() #Parse the arguments
parser.add_argument('--run_id', type=str, default='run_1') #Run id
parser.add_argument('--dataset', type=str, default='trivia_qa') #Dataset name
parser.add_argument('--project', type=str, default='opt_350m') #Project name
parser.add_argument('--model', type=str, default='opt-350m') #Model name
args = parser.parse_args() #Parse the arguments

# Load rouge metric
rouge_score = load('rouge')

def get_similarities_df():
    """Get the similarities df from the pickle file"""
    with open(f'{config.output_dir}/semantic_similarities/{args.run_id}/{args.dataset}_{args.model}_generations_similarities.pkl', 'rb') as f:
        similarities = pickle.load(f) #Load the semantic similarities from the pickle file. This is a nested dictionary with keys as the ids and values as the semantic similarities.
        similarities_df = pd.DataFrame.from_dict(similarities, orient='index') #Convert the similarities dictionary to a pandas dataframe. index is the id and the values are the semantic similarities.
        similarities_df['id'] = similarities_df.index #Set the id to the index
        similarities_df['has_semantically_different_answers'] = similarities_df[
            'has_semantically_different_answers'].astype('int') #Set the has semantically different answers to the integer value
        similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
            lambda x: x['rougeL']) #Set the rougeL among generations to the rougeL value

        return similarities_df

def get_generations_df():
        """Get the generations df from the pickle file"""
        with open(f'{config.output_dir}/sequences/{args.run_id}/{args.dataset}_{args.model}_generations.pkl', 'rb') as infile:
            generations = pickle.load(infile) #Load the generations from the pickle file.
            generations_df = pd.DataFrame(generations) #Convert the generations to a pandas dataframe.
            generations_df['id'] = generations_df['id'].apply(lambda x: x[0]) #Extract the id.
            generations_df['id'] = generations_df['id'].astype('object') #Set the id to the object type.
            generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int') #Set the correct to the integer value.

            return generations_df

def get_likelihoods_df():
    """Get the likelihoods df from the pickle file"""

    with open(f'{config.output_dir}/confidence_measures/{args.run_id}/{args.dataset}_aggregated_likelihoods_{args.model}_generations.pkl', 'rb') as f:
        likelihoods = pickle.load(f) #Load the likelihoods from the pickle file.

        keys_to_use = ('ids', 'predictive_entropy', 'average_predictive_entropy',\
                        'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts')

        likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use)
        for key in likelihoods_small:
            if type(likelihoods_small[key]) is torch.Tensor:
                likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())

        likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

        likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

        return likelihoods_df

def calculate_accuracy(result_df, threshold, column_name):
    """Calculate the accuracy for a given threshold and column name"""
    accuracy = result_df[result_df[column_name] > threshold]['correct'].mean()
    return accuracy

def get_best_threshold_for_accuracy(result_df):
    """Get the best threshold for accuracy"""
    #get a copy of the unnormalised_entropy_over_concepts and correct columns df together
    df = result_df[['unnormalised_entropy_over_concepts', 'correct']].copy()
    #sort the df by the unnormalised_entropy_over_concepts column
    df = df.sort_values(by='unnormalised_entropy_over_concepts')
    #list of values between 0.05 to 0.95 in steps of 0.05
    values = np.arange(0.05, 0.95, 0.05)
    thresholds = []
    for value in values:
        #find out the entropy value that is value percentile of the unnormalised_entropy_over_concepts column
        threshold = df['unnormalised_entropy_over_concepts'].quantile(value)
        thresholds.append(threshold)

    best_threshold = None
    best_accuracy = 0
    for threshold in thresholds:
        accuracy = calculate_accuracy(result_df, threshold, 'unnormalised_entropy_over_concepts')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def get_false_negatives(result_df, threshold):
    """Get the false negatives for a given threshold"""
    #copy the result_df
    df = result_df.copy()
    #if the unnormalised_entropy_over_concepts is greater than the threshold, set the prediction_based_on_semantic_similarity to 1, otherwise set it to 0
    df["prediction_based_on_semantic_similarity"] = df['unnormalised_entropy_over_concepts'].apply(lambda x: 0 if x > threshold else 1)
    # get only those rows where the correct is 1 and prediction_based_on_semantic_similarity is 0
    false_negatives_df = df[(df['correct'] == 1) & (df['prediction_based_on_semantic_similarity'] == 0)]
    #reindex the false_negatives
    false_negatives_df = false_negatives_df.reset_index(drop=True)
    return false_negatives_df

def get_true_negatives(result_df, threshold):
    """Get the true negatives for a given threshold"""
    #copy the result_df
    df = result_df.copy()
    #if the unnormalised_entropy_over_concepts is greater than the threshold, set the prediction_based_on_semantic_similarity to 1, otherwise set it to 0
    df["prediction_based_on_semantic_similarity"] = df['unnormalised_entropy_over_concepts'].apply(lambda x: 0 if x > threshold else 1)
    # get only those rows where the correct is 0 and prediction_based_on_semantic_similarity is 0
    true_negatives_df = df[(df['correct'] == 0) & (df['prediction_based_on_semantic_similarity'] == 0)]
    #reindex the true_negatives
    true_negatives_df = true_negatives_df.reset_index(drop=True)
    return true_negatives_df

def average_num_clusters(df):
    """Average the number of clusters"""
    #for each row in the df, get the number of unique values in the semantic_set_ids column
    num_clusters = df['semantic_set_ids'].apply(lambda x: len(set(x)))
    #return the average of the number of clusters
    return num_clusters.mean()

def find_correct_cluster_id(df):
    """Find the correct cluster for a given df"""
    #copy the df
    df = df.copy()
    #add a new column "correct_cluster_id" to the df and initialize it to -1
    df["correct_cluster_id"] = -1
    #for each row in the df, find the cluster id that is the most similar to the answer
    for index, row in df.iterrows():
        #get the answer
        answer = row["answer"]
        cluster_ids = set(row["semantic_set_ids"])
        rougeL_scores = {}
        for cluster_id in cluster_ids:
            rougeL_scores[cluster_id] = 0

        generation_texts = row["generated_texts"]
        for cluster_id in cluster_ids:
            #get the generations_text for the cluster_id
            generations_text = generation_texts[row["semantic_set_ids"] == cluster_id]
            cluster_rougeL_score = 0
            for generation_text in generations_text:
                rougeL_score = rouge_score.compute(predictions=[generation_text], references=[answer])["rougeL"]
                cluster_rougeL_score += rougeL_score
            cluster_rougeL_score /= len(generations_text)
            rougeL_scores[cluster_id] = cluster_rougeL_score
        correct_cluster_id = max(rougeL_scores, key=rougeL_scores.get)
        if rougeL_scores[correct_cluster_id] > 0.3:
            df.at[index, "correct_cluster_id"] = correct_cluster_id
    return df
def find_avg_generations_correct_cluster(df):
    """Find the average number of generations for the correct cluster"""
    #iterate over each row
    no_of_generations = 0
    count_of_rows = 0
    for index, row in df.iterrows():
        #get the correct cluster id
        correct_cluster_id = row["correct_cluster_id"]
        if correct_cluster_id != -1:
            #get the generations_text for the correct cluster id
            generations_text = row["generated_texts"][row["semantic_set_ids"] == correct_cluster_id]
            no_of_generations += len(generations_text)
            count_of_rows += 1
    if count_of_rows > 0:
        return no_of_generations / count_of_rows
    else:
        return 0

def find_avg_generations_incorrect_clusters(df):
    """Find the average number of generations for the incorrect clusters"""
    #iterate over each row
    no_of_generations = []
    for index, row in df.iterrows():
        #get the correct cluster id
        correct_cluster_id = row["correct_cluster_id"]
        if correct_cluster_id == -1:
            semantic_set_ids = set(row["semantic_set_ids"])
            for cluster_id in semantic_set_ids:
                if cluster_id != correct_cluster_id:
                    generations_text = row["generated_texts"][row["semantic_set_ids"] == cluster_id]
                    no_of_generations.append(len(generations_text))
        else:
            for cluster_id in semantic_set_ids:
                generations_text = row["generated_texts"][row["semantic_set_ids"] == cluster_id]
                no_of_generations.append(len(generations_text))
    
    if len(no_of_generations) > 0:
        return sum(no_of_generations) / len(no_of_generations)
    else:
        return 0

similarity_df = get_similarities_df()
generations_df = get_generations_df()
likelihoods_df = get_likelihoods_df()

result_df = similarity_df.merge(generations_df, on='id').merge(likelihoods_df, on='id')

best_threshold, best_accuracy = get_best_threshold_for_accuracy(result_df)
print(f'Best threshold for accuracy: {best_threshold}, Best accuracy: {best_accuracy}')

false_negatives_df = get_false_negatives(result_df, best_threshold)
true_negatives_df = get_true_negatives(result_df, best_threshold)
false_negatives_avg_clusters = average_num_clusters(false_negatives_df)
true_negatives_avg_clusters = average_num_clusters(true_negatives_df)
print(f'Number of false negatives: {len(false_negatives_df)}')
print(f'Number of true negatives: {len(true_negatives_df)}')
print(f'False negatives average number of clusters: {false_negatives_avg_clusters}')
print(f'True negatives average number of clusters: {true_negatives_avg_clusters}')
#also print whether field "answer" is present in result_df or not. 
print(f'Answer field present in result_df: {"answer" in result_df.columns}')
if "answer" in result_df.columns:
    print(f'Answer field present in result_df: {result_df["answer"].head()}')
else:
    print("Answer field not present in result_df")

correct_cluster_id_df = find_correct_cluster_id(result_df)
avg_generations_correct_cluster = find_avg_generations_correct_cluster(correct_cluster_id_df)
avg_generations_incorrect_clusters = find_avg_generations_incorrect_clusters(correct_cluster_id_df)
print(f'Average number of generations for the correct cluster: {avg_generations_correct_cluster}')
print(f'Average number of generations for the incorrect clusters: {avg_generations_incorrect_clusters}')
