import pandas as pd

pd.options.display.max_colwidth = 100
df = pd.read_json('data.jsonl', lines=True)

for i, v in df.iterrows():
    if i == 356:
        print(v.prompt)