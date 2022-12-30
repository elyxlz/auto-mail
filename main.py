import pandas as pd
import numpy as np
from datasets import Dataset
import re
import json
import argparse
import dotenv
import os
import time
import sys

dotenv.load_dotenv(override=True)

parser = argparse.ArgumentParser(description='Data Module')
parser.add_argument("--path", help="Path to email.csv file", required=True)
parser.add_argument("--name", help="Number of emails to train on", required=True)
parser.add_argument("--num_workers", help="Number of CPU logical processors", default='1')
parser.add_argument("--model", help="Model to use, either curie or davinci", default='curie')
parser.add_argument("--data_amount", help="Number of emails to train on", default='1000')
parser.add_argument("--num_epochs", help="Number of emails to train on", default='2')


costs = {
    'curie' : 0.003,
    'davinci' : 0.03,
}

args = parser.parse_args()
path = args.path
name = args.name
num_proc = int(args.num_workers)
model = args.model
data_amount = int(args.data_amount)
num_epochs = int(args.num_epochs)

def preprocess(row):
    """
    Function that cleans the text and identifies the prompt / completion sections out of the body
    Ignores all emails that are not replies to another email or are empty forwards
    """
    text = row['Body']
    text = re.sub('\r\n', '\n', str(text))
    reg = re.findall(r'((?:.|\n)*?)((?:From|Da): .*?\n(?:.|\n)*?(?:Subject|Oggetto): .*?)\n(?:\n|\s)+(.*?(?:.|\n)*)', text)

    if len(reg) > 0:
        reg = reg[0]
        Sent = reg[0]
        Info = reg[1]
        Received = reg[2]

        prompt = f"{Info}\n\n{Received}"
        prompt = re.sub('###', '', prompt)
        prompt = prompt[:2000*4]
        prompt += '###'
        
        completion = f" {Sent}"
        completion = re.sub('###', '', completion)
        completion = completion[:2000*4]
        completion += '###'
        if len(prompt + completion) > 2047 * 4 or completion.isspace() or 'FYI' in completion: prompt,completion=None,None # remove overtly long ones

    else: prompt,completion=None,None # if the email is not a reply
        
    return dict(prompt=prompt, completion=completion)

print("Loading Data")
df = pd.read_csv(path)[['Subject', 'Body', 'From: (Name)','To: (Name)', 'CC: (Name)', 'BCC: (Name)']]
df.rename(columns={'From: (Name)' : 'Account', 'To: (Name)' : 'To', 'CC: (Name)' : 'CC', 'BCC: (Name)' : 'BCC'}, inplace=True)
print("Processing Data")
df = pd.DataFrame(Dataset.from_pandas(df).map(preprocess, num_proc=num_proc)).dropna()[['prompt', 'completion']]
df = df.iloc[np.random.choice(len(df), data_amount)]

print("Saving Data")
df.to_json('data.jsonl', orient='records', lines=True)

def find_length(cell): return len(cell)
total = df.applymap(find_length).sum().sum()
print(f"\n\n\n\n\n\n\n\n\n\nApproximated credit costs: {total / 4 / 1000 * num_epochs * costs[model]}\n\n$")

time.sleep(5)
#os.system('openai tools fine_tunes.prepare_data -f data.json')

inp = input("\n\nProceed with fine-tuning? [y/n]")
if inp == 'n': sys.exit()

os.system(f'openai api fine_tunes.create -t data.jsonl -m {model} --n_epochs {num_epochs} --learning_rate_multiplier 0.1 --batch_size 32 --suffix "{name}"')
print('yoo')