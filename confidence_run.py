
import json
import pandas as pd
import random
random.seed(42)
import os
from datasets import load_dataset
from argparse import ArgumentParser
from openai import OpenAI
from prompts import prompts, deepseek_extra_prompt
from tqdm import tqdm

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
            question = problem + '\n' + choices_str
            yield {'question': question, 'choices': choices_str, 'gold_answer': gold_answer, 'gold_option': gold_option, 'idx': idx}
    elif dataset_name == 'gpqa_free':
        data = load_dataset('nikhilchandak/freeform-datasets', split='gpqa_diamond')
        data = pd.DataFrame(data)
        for idx, row in data.iterrows():
            problem = row['question']
            gold_answer = row['answer'].lower()
            yield {'question': problem,'gold_answer': gold_answer, 'idx': idx}
    elif dataset_name == 'hle_mcq':
        d = pd.read_json('/home/palipoormola/uncertainty/hle_mcq.jsonl', lines=True)
        for idx, row in d.iterrows():
            question = row['question']
            answer = row['answer']
            gold_option = answer
            yield {'question': question, 'gold_answer': answer, 'gold_option': gold_option, 'idx': idx}
    elif dataset_name == 'hle':
        data = load_dataset('cais/hle')
        data = data['test'].to_pandas()
        data = data[data['image'] == '']
        data = data[data['rationale_image'].isnull()].reset_index(drop=True)
        data = data[data['answer_type'] == 'exactMatch'].reset_index(drop=True)
        print(f"Total samples in HLE after filtering: {len(data)}")
        for idx, row in data.iterrows():
            question = row['question']
            answer = row['answer']
            gold_option = answer
            yield {'question': question, 'gold_answer': answer, 'gold_option': answer , 'idx': idx}

    elif dataset_name == 'simpleqa': 
        data = load_dataset('google/simpleqa-verified')
        data = data['eval'].to_pandas()
        for idx, row in data.iterrows():
            question = row['problem']
            answer = row['answer']
            gold_option = answer
            yield {'question': question, 'gold_answer': answer, 'gold_option': answer , 'idx': idx}
    elif dataset_name == 'livebench':
        data = load_dataset('livebench/reasoning', split='test')
        data = data.to_pandas()
        for idx, row in data.iterrows():
            question = row['turns'][0].split('Think step by step')[0].strip()
            answer = row['ground_truth']
            gold_option = answer
            yield {'question': question, 'gold_answer': answer, 'gold_option': answer , 'idx': idx}
    elif dataset_name == 'diagnosisArena':
        data = load_dataset('shzyk/DiagnosisArena', split='test')
        data = data.to_pandas()
        for idx, row in data.iterrows():
            case_info = row['Case Information']
            physical_examination = row['Physical Examination']
            diagnostic_tests = row['Diagnostic Tests']
            answer_choices = row['Options']
            answer_choices_str = '\n'.join([k + ') ' + v for k,v in answer_choices.items()])
            gold_option = row['Right Option']
            question = f"""Case Information: {case_info}\nPhysical Examination: {physical_examination}\nDiagnostic Tests: {diagnostic_tests}\nBased on the above information, what is the most likely diagnosis?\nAnswer Choices: {answer_choices_str}\n"""
            yield {'question': question, 'gold_answer': gold_option, 'gold_option': gold_option , 'idx': idx}
    elif 'mmlupro' in dataset_name:
        data_file = '/home/palipoormola/uncertainty/early_results/mmlu_pro_questions.jsonl'
        data = pd.read_json(data_file, lines=True)
        for idx, row in data.iterrows():
            problem = row['question']
            symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            options = row['options']
            choices_str = '\n'.join([symbols[i] + ')' + options[i] for i in range(0, len(options))])
            gold_option = row['answer']
            gold_answer = options[symbols.index(gold_option)]
            if 'free' in dataset_name:
                question = problem
            else:
                question = problem + '\n' + choices_str
            yield {'question': question, 'gold_answer': gold_answer, 'gold_option': gold_option, 'idx': idx}
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")
    

def llm_invoke_wo_sysprompt(client, model, prompt, sys_prompt, sampling_params, n=1, logprobs=True, nlogprobs=10):
    extra_body = {}
    if 'min_p' in sampling_params:
        extra_body['min_p'] = sampling_params['min_p']
    if 'top_k' in sampling_params:
        extra_body['top_k'] = sampling_params['top_k']
    if type(client) == OpenAI:
        chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": sys_prompt + '\n' + prompt}],
        logprobs=logprobs,
        top_logprobs=nlogprobs if logprobs else 0,
        temperature= sampling_params.get('temperature', 0.0),
        top_p=sampling_params['top_p'],
        n = n,
        max_tokens=sampling_params.get('max_tokens', 15000),
        extra_body=extra_body,
        )
        return chat_response
    else:
        raise NotImplementedError("This function is only implemented for OpenAI client currently.")
    
def llm_invoke(client, model, prompt, sys_prompt, sampling_params, n=1, logprobs=True, nlogprobs=10, api_provider='together'):
    extra_body = {}
    if 'min_p' in sampling_params:
        extra_body['min_p'] = sampling_params['min_p']
    if 'top_k' in sampling_params:
        extra_body['top_k'] = sampling_params['top_k']
    
    if 'deepseek' in model and api_provider == 'together':
        chat_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}],
            logprobs=nlogprobs if logprobs else 0,
            temperature= sampling_params.get('temperature', 0.0),
            top_p=sampling_params['top_p'],
            n = n,
            max_tokens=sampling_params.get('max_tokens', 15000),
            extra_body=extra_body,
            reasoning={"enabled": True},
        )
    elif 'deepseek' in model and api_provider == 'deepseek':
        chat_response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            logprobs=logprobs,
            top_logprobs=nlogprobs if logprobs else 0,
            temperature=sampling_params.get('temperature', 0.0),
            top_p=sampling_params['top_p'],
            max_tokens=sampling_params.get('max_tokens', 15000),
            extra_body={"thinking": {"type": "enabled"}, 'top_k': 20}
        )
    elif 'deepseek' in model and api_provider == 'hf':
        chat_response = client.chat.completions.create(
            model=model+ ':fireworks-ai',
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": prompt}],
            logprobs=logprobs,
            top_logprobs=min(nlogprobs,5) if logprobs else 0,
            temperature= sampling_params.get('temperature', 0.0),
            top_p=sampling_params['top_p'],
            n = n,
            stream = False,
            max_tokens=sampling_params.get('max_tokens', 15000),
            extra_body=extra_body,
            reasoning_effort="high",
        )
    else:
        if type(client) == OpenAI:
            if api_provider == 'hf':
                model = model + ':fireworks-ai'
            if 'repetition_penalty' in sampling_params:
                extra_body['repetition_penalty'] = sampling_params['repetition_penalty']
            chat_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}],
            logprobs=logprobs,
            top_logprobs=nlogprobs if logprobs else 0,
            temperature= sampling_params.get('temperature', 0.0),
            top_p=sampling_params['top_p'],
            n = n,
            max_tokens=sampling_params.get('max_tokens', 15000),
            extra_body=extra_body,
            )
        else:
            chat_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}],
            logprobs=nlogprobs if logprobs else False,
            temperature= sampling_params.get('temperature', 0.0),
            top_p=sampling_params['top_p'],
            n = n,
            max_tokens=sampling_params.get('max_tokens', 15000),
            extra_body=extra_body,
            )
    return chat_response

def convert_logprobs_to_standard_format(logprobs):
    content_logprobs = logprobs['content']
    tokens_list = []
    logprobs_list = []
    top_logprobs_list = []
    if content_logprobs is None:
        return {
            'tokens': None,
            'logprobs': None,
            'top_logprobs': None
        }
    for i, x in enumerate(content_logprobs):
        tlp_dict = {}
        tokens_list.append(x['token'])
        logprobs_list.append(x['logprob'])
        for tlp in x['top_logprobs']:
            tlp_dict[tlp['token']] = tlp['logprob']
        top_logprobs_list.append(tlp_dict)
    return {
        'tokens': tokens_list,
        'logprobs': logprobs_list,
        'top_logprobs': top_logprobs_list
    }

def main(args):
    
    model = args.model
    with open(args.key_file, 'r') as f:
        try:
            api_keys = json.load(f)
        except:
            raise ValueError(f"Failed to load API keys from {args.key_file}. Ensure it's a valid JSON file.")
    with open(args.sampling_params, 'r') as f:
        try:
            sampling_params = json.load(f)
        except:
            raise ValueError(f"Failed to load sampling parameters from {args.sampling_params}. Ensure it's a valid JSON file.")
    prompt = prompts.get(args.prompt, None)
    if prompt is None:
        raise ValueError(f"Prompt key {args.prompt} not found in prompts dictionary.")
    
    results_dir = os.path.join(args.result_dir, f"{args.dataset}{args.version_suffix}/{model.replace('/','_')}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump({
            'arguments': vars(args),
            'sampling_params': sampling_params,
            'prompt': prompt,
        }, f, indent=4)

    if args.api_provider == 'hf':
        os.environ["HF_TOKEN"] = api_keys['hf']
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            timeout=300
        )
    elif args.api_provider == 'wandb':
        client = OpenAI(
            base_url='https://api.inference.wandb.ai/v1',
            api_key=api_keys['wandb'],
            timeout=180)
    elif args.api_provider == 'together':
        from together import Together
        os.environ["TOGETHER_API_KEY"] = api_keys['together']
        client = Together(timeout=300)
    elif args.api_provider == 'bedrock':
        raise NotImplementedError("Bedrock API not implemented yet.")
    elif args.api_provider == 'tinker':
        raise NotImplementedError("Tinker API not implemented yet.")
        # import tinker
        # service_client = tinker.ServiceClient(api_keys['tinker'])
        # client = service_client.create_sampling_client(base_model = model)
    elif args.api_provider == 'vllm':
        client = OpenAI(
            api_key='baaa',
            base_url=f"http://localhost:{args.port}/v1",
            timeout = 180
        )
    elif args.api_provider == 'deepseek':
        os.environ["DEEPSEEK_API_KEY"] = api_keys['deepseek']
        client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    else:
        raise ValueError(f"Unknown API provider: {args.api_provider}")
    dataset = args.dataset
    for data_row in tqdm(get_next_data_row(dataset)):
        if data_row['idx'] < args.start_idx:
            continue
        if args.num_samples != -1 and data_row['idx'] >= args.start_idx + args.num_samples:
            break
        question = data_row['question']
        gold_answer = data_row['gold_answer']
        gold_option = data_row.get('gold_option', None)
        res = {
            'question': question,
            "gold_answer": gold_answer,
            "gold_option": gold_option if gold_option != gold_answer else None,
        }
        result_file = os.path.join(results_dir, f'result_{data_row["idx"]}.json')
        try:
            with open(result_file, 'r') as f:
                existing_res = json.load(f)
                if existing_res.get('verbalized_confidence', 1.5) == 0:
                    raise ValueError("Existing result has confidence 0, re-running the sample.")
            continue
        except:
            pass
        try:
            if 'R1' in model:
                response = llm_invoke_wo_sysprompt(client, model, question, prompt, sampling_params, n=1, logprobs=args.lp, nlogprobs=args.nlogprobs)
            else:
                if 'deepseek' in model:
                    prompt = prompt + deepseek_extra_prompt
                response = llm_invoke(client, model, question, prompt, sampling_params, n=1, logprobs=args.lp, nlogprobs=args.nlogprobs, api_provider=args.api_provider)
        except Exception as e:
            print("Error during LLM invocation:", e)
            continue
        resp_choice = response.choices[0]
        resp_dict = resp_choice.model_dump()
        if 'logprobs' in resp_dict and resp_dict['logprobs'] is not None:
            if 'content' in resp_dict['logprobs']:
                resp_dict['logprobs'] = convert_logprobs_to_standard_format(resp_dict['logprobs'])
        else:
            resp_dict['logprobs'] = None
        res['response'] = resp_dict
        with open(result_file, 'w') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)


argparser = ArgumentParser()
argparser.add_argument('--dataset', type=str)
argparser.add_argument('--model', type=str)
argparser.add_argument('--num_samples', type=int, default=-1)
argparser.add_argument('--port', type=int, default=8000)
argparser.add_argument('--result_dir', type=str, default='./confidence_results_march')
argparser.add_argument('--sampling_params', type=str, help='Filepath to JSON containing sampling parameters like temperature, top_p, etc.')
argparser.add_argument('--prompt', type=str, help='Key to prompt template to use.')
argparser.add_argument('--api_provider', type=str, choices=['hf', 'together', 'bedrock', 'tinker', 'vllm', 'wandb', 'deepseek'])
argparser.add_argument('--key_file', type=str, help='Filepath to JSON containing API keys and other necessary info for the chosen API provider', default='api_keys.json')
argparser.add_argument('--nlogprobs', type=int, default=10, help='Number of logprobs to return from the model.')
argparser.add_argument('--lp', action='store_true', help='Whether to get logprobs or not.')
argparser.add_argument('--start_idx', type=int, default=0, help='Starting index of the dataset to process.')
argparser.add_argument('--version_suffix', type=str, default='')
if __name__ == "__main__":
    args = argparser.parse_args()
    with open(args.sampling_params, 'r') as f:
        try:
            sampling_params = json.load(f)
        except:
            raise ValueError(f"Failed to load sampling parameters from {args.sampling_params}. Ensure it's a valid JSON file.")
    
    main(args)


"""
model names to copy :D 

Qwen/Qwen3-235B-A22B-Thinking
deepseek-ai/DeepSeek-V3.1
openai/gpt-oss-120b
openai/gpt-oss-20b
Qwen/Qwen3-32B

"""