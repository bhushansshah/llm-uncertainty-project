
from argparse import ArgumentParser
from unittest import result
from openai import OpenAI
import pandas as pd
import random
random.seed(42)
import os
from datasets import load_dataset

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

qwen_prompt = """You will be given a question. Answer the question by choosing one of the provided options.
Provide the final answer in the last line, prefixed with "Answer:". Do not answer with a full sentence. Just provide the letter of the correct choice.

Here is the question:
{question}
{choices}

<think>
"""
phi_prompt = """ Answer the question by choosing one of the provided options.
Provide the final answer in the last line, prefixed with "Answer:". Do not answer with a full sentence. Just provide the letter of the correct choice, like: Answer: A

Here is the question:
{question}
{choices}
"""

mof_qwen_prompt = """ Your task is to consider a scientific claim, and determine whether how it is true/feasible, or false/infeasible. You must also provide a brief explanation as to why you consider the claim more likely to be true/feasible or false/infeasible.
If you consider the claim more likely to be true/feasible, then you should output True. If you consider the claim more likely to be false/infeasible, then you should output False.
Think step by step to arrive at the final answer.
Provide your reasoning steps clearly. Answer when you are done reasoning. Do not ramble

Your output must be in the following format:
A dictionary with 2 main keys: `claim_true_or_false` and `explanation`.
The value corresponding to the key `claim_true_or_false` must be either True or False.
The value corresponding to the key `explanation` must be a brief explanation (1-3 sentences) as to why you consider the claim more likely to be true/feasible or false/infeasible.

Your answer json must be the final output, and nothing should follow after it.

Here is the claim to evaluate:
{claim_text}

"""

mof_phi_prompt = mof_qwen_prompt

QWEN_DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "min_p": 0,
    "top_k": 20,
    "do_sample": True
} 

PHI_DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "do_sample": True
}

CONTROLLED_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True
}
    
def get_full_question(dataset_name, data_row):
    if dataset_name in ['gpqa', 'mmlupro']:
        prompt = phi_prompt if 'Phi' in args.model else qwen_prompt
        return prompt.replace("{question}", data_row['question']).replace("{choices}", data_row['choices'])
    elif dataset_name == 'matteroffact':
        prompt = mof_qwen_prompt if 'Qwen' in args.model else mof_phi_prompt
        return prompt.replace("{claim_text}", data_row['claim_text'])
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")
        
def get_next_data_row(dataset_name):
    if dataset_name == 'gpqa':
        data = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        data = data['train'].to_pandas()
        for idx, row in data.iterrows():
            problem = row['Question']
            gold_answer = row['Correct Answer'].lower()
            options = ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
            random.shuffle(options)
            symbols = ['A)', 'B)', 'C)', 'D)']
            choices_str = '\n'.join([symbols[i] + row[options[i]] for i in range(0,4)])
            gold_option = symbols[options.index('Correct Answer')].replace(')','')
            yield {'question': problem, 'choices': choices_str, 'gold_answer': gold_answer, 'gold_option': gold_option, 'idx': idx}
    elif dataset_name == 'mmlupro':
        data_file = '../../mmlu_pro_questions.jsonl'
        data = pd.read_json(data_file, lines=True)
        for idx, row in data.iterrows():
            problem = row['question']
            symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            options = row['options']
            choices_str = '\n'.join([symbols[i] + ')' + options[i] for i in range(0, len(options))])
            gold_option = row['answer']
            gold_answer = options[symbols.index(gold_option)]
            yield {'question': problem, 'choices': choices_str, 'gold_answer': gold_answer, 'gold_option': gold_option, 'idx': idx}
    elif dataset_name == 'matteroffact':
        data = pd.read_json('../matteroffact_subset.json', orient='records')
        for idx, row in data.iterrows():
            gold_answer = row['gold_label']
            claim_text = row['claim_text']
            yield {'claim_text': claim_text, 'gold_answer': gold_answer, 'idx': idx}
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

def main(args):
    model = args.model
    if args.controlled_sampling:
        sampling_params = CONTROLLED_SAMPLING_PARAMS
    else:
        if 'Qwen' in model:
            sampling_params = QWEN_DEFAULT_SAMPLING_PARAMS
        elif 'Phi' in model:
            sampling_params = PHI_DEFAULT_SAMPLING_PARAMS
        else:
            raise NotImplementedError(f"Model {model} not implemented.")
            
    if args.port != 8000:
        openai_api_base = f"http://localhost:{args.port}/v1"
    else:
        openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )
    def llm_invoke(prompt, n=1, logprobs=True, nlog_probs=10, override_temperature=None):
        extra_body = {}
        if 'min_p' in sampling_params:
            extra_body['min_p'] = sampling_params['min_p']
        if 'top_k' in sampling_params:
            extra_body['top_k'] = sampling_params['top_k']
        chat_response = client.completions.create(
            model=model,
            prompt = prompt,
            temperature= override_temperature if override_temperature is not None else sampling_params['temperature'], 
            top_p=sampling_params['top_p'],
            extra_body=extra_body,
            logprobs=nlog_probs if logprobs else 0,
            n = n,
            max_tokens=15000
        )

        return chat_response
    
    root_dir = os.path.join(args.result_dir, f"{args.dataset}_{args.model.replace('/','_')}")
    if args.temp is not None:
        root_dir = os.path.join(root_dir, f"temp_{args.temp}")

    import json
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, 'config.json'), 'w') as f:
        json.dump({
            'arguments': vars(args),
            'prompt': qwen_prompt if 'Qwen' in args.model else phi_prompt,
            'sampling_params': sampling_params
        },f, indent=4)
        
    for data_row in get_next_data_row(args.dataset):
        if args.num_samples != -1 and data_row['idx'] >= args.num_samples:
            break
        print(f"Processing idx: {data_row['idx']}")
        question = get_full_question(args.dataset, data_row)
        res = {
            'question': question,
            "gold answer": data_row['gold_answer'],
        }
        if 'gold_option' in data_row:
            res['gold option'] = data_row['gold_option']
        
        response = llm_invoke(question, n=1, logprobs=True, nlog_probs=10, override_temperature=args.temp)
        resp_choice = response.choices[0]
        resp_dict = resp_choice.model_dump()
        resp_dict['logprobs'].pop('text_offset', None)
        res['response'] = resp_dict
        result_file = os.path.join(root_dir, f'result_{data_row["idx"]}.json')
        with open(result_file, 'w') as f:
            json.dump(res, f, indent=4)
    
argparser = ArgumentParser()
argparser.add_argument('--dataset', type=str, choices=['gpqa', 'mmlupro', 'matteroffact'], default='gpqa')
argparser.add_argument('--model', type=str, choices=['Qwen/Qwen3-32B', 'Qwen/Qwen3-14B', 'microsoft/Phi-4-reasoning', 'ernie', 'deepseek_llama'], default='Qwen/Qwen3-14B')
argparser.add_argument('--num_samples', type=int, default=-1)
argparser.add_argument('--port', type=int, default=8000)
argparser.add_argument('--result_dir', type=str, default='./single_trajs/')
argparser.add_argument('--controlled_sampling', action='store_true', help='Use controlled sampling parameters.')
argparser.add_argument('--temp', type=float, default=None, help='Override temperature setting.')
if __name__ == '__main__':
    args = argparser.parse_args()
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    if args.model == 'ernie':
        args.model = 'baidu/ERNIE-4.5-21B-A3B-Thinking'
    if args.model == 'deepseek_llama':
        args.model= 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    main(args)
