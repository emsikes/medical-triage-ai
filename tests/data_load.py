import os
import json


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

load_data = load_jsonl('data/processed/train.jsonl')
print(load_data)
