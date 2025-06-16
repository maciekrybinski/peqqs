import pandas as pd
from langchain_community.llms import VLLM
from transformers import AutoTokenizer
import torch
import argparse
import json
from utils import model_dict, parse_topics

topics = parse_topics('../data/topics.xml')[0]
topics = {top['id']:top for top in topics}
doc_dict = json.load(open('../data/docs.json', 'r'))
ground_truth = json.load(open('../data/quantities_ground_truth.json', 'r'))

irrelevant_paper_list = {}
with open('../data/10_neg_samples.txt', 'r') as file:
    lines = [line.strip() for line in file if line.strip()] 
    for i in range(0, len(lines), 2):
        key = int(lines[i])
        values = set(map(int, lines[i + 1].strip('{}').split(', ')))
        irrelevant_paper_list[key] = values

message_template = (
    "Return a ';'-separated list of quantities that represent a specific absolute value of {variable} for {crop} "
    "in the following scientific abstract: \n"
    "Title: {title} \n"
    "Abstract: {abstract} \n"
    "Return 'NONE', if no values of {variable} for {crop} are reported. "
    "You must respond the results as a semicolon-separated list in the format: value unit; value unit; value unit. "
    "Do not include any explanations or additional text."
)

def run_inference(tokenizer, paper_id, topic_id, variable, crop, title, abstract, llm):
    current_message_content = message_template.format(variable=variable, crop=crop, title=title, abstract=abstract)
    messages = [{"role": "user", "content": current_message_content}]
    try:
        output = llm.invoke(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception as e:
        print(f"Error processing row {e}, {paper_id}, {topic_id}")
        output = ''
    print(output)
    return {'category': topic_id, 'crop':crop, 'variable': variable, 'paper_id': paper_id, 'title': title, 'outputs': output}

def run_exp1(model_name, save_path, exp_num):
    model_path = model_dict[model_name]
    if model_path == '':
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loading model: {model_name}")

    llm = VLLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(), temperature=0,
        trust_remote_code=True,
        vllm_kwargs={
            "gpu_memory_utilization": 0.9,
            "max_model_len": 15000,
        },
    )

    for i in range(1, 4):
        result = []        
        if exp_num == 2:
            for k in irrelevant_paper_list.keys():
                topic_id = k
                selected_papers = list(irrelevant_paper_list[topic_id])
                for paper_id in selected_papers:     
                    variable, crop = topics[str(topic_id)]['param'].replace('"', '').strip(), topics[str(topic_id)]['crop']
                    title = doc_dict[str(paper_id)]['title'].strip()
                    abstract = doc_dict[str(paper_id)]['abstract'].strip()
                    result.append(run_inference(tokenizer, paper_id, topic_id, variable, crop, title, abstract, llm))

        if exp_num == 1 or exp_num == 3:            
            for item in ground_truth:
                paper_id, topic_id = item['doc_id'], item['topic']
                title, abstract = doc_dict[paper_id]['title'].strip(), doc_dict[paper_id]['abstract'].strip()
                variable, crop = topics[topic_id]['param'].replace('"', '').strip(), topics[topic_id]['crop']
                print(topic_id, paper_id, variable, crop, paper_id, title)
                    
                if exp_num == 3:
                    selected_papers = list(irrelevant_paper_list[int(topic_id)])[:3]
                    print(f"tpoic_id: {topic_id}, irrelevant papers: {selected_papers}")
                    for item in selected_papers:
                        title += doc_dict[str(item)]['title'].strip()
                        abstract += doc_dict[str(item)]['abstract'].strip()
                result.append(run_inference(tokenizer, paper_id, topic_id, variable, crop, title, abstract, llm))

        df_records = pd.DataFrame(result)
        df_records.to_csv(f"{save_path}{model_name.split('/')[-1]}_exp_{exp_num}_{i}.csv", index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Model name')
    parser.add_argument('--save_path', type=str, default='./', help='Save path')
    parser.add_argument('--exp_num', type=int, default=1, help='Experiment number, 1, 2 or 3')
    args = parser.parse_args()
    run_exp1(args.model_name, args.save_path, args.exp_num)
