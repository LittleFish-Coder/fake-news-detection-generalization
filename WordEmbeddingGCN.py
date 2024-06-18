# import package
import numpy as np
import pandas as pd
import torch
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep="\t", encoding="utf-8")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = self.df.iloc[idx]["label"]
        wrapped_input = tokenizer(text, max_length=16, add_special_tokens=True, truncation=True, padding="max_length", return_tensors="pt")
        return wrapped_input, label


csv_file = "dataset/train.csv"
dataset = TextDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
embeddings_ls = []
labels_ls = []
for batch in tqdm(dataloader, desc="Text to tokenize encoding vector to word embedding..."):
    wrapped_input_dics, labels = batch
    labels_ls.append(labels)
    # print(wrapped_input_dics.keys())
    # print(len(wrapped_input_dics['input_ids']))
    # print(len(wrapped_input_dics['token_type_ids']))
    # print(len(wrapped_input_dics['attention_mask']))
    input_ids = wrapped_input_dics["input_ids"].squeeze(1)
    attention_mask = wrapped_input_dics["attention_mask"].squeeze(1)
    token_type_ids = wrapped_input_dics["token_type_ids"].squeeze(1)
    # 将输入特征传递给 BERT 模型
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # 获取最后一层隐藏状态（即嵌入向量）
    embeddings = outputs.last_hidden_state
    # 对所有标记的嵌入向量求平均
    sentence_embeddings = torch.mean(embeddings, dim=1)
    embeddings_ls.append(sentence_embeddings)
    # print(embeddings)
    # print(labels)      # 这些是标签
print(len(embeddings_ls))
print(len(labels_ls))

all_text_embeddings = []
all_labels = []
for batch in tqdm(embeddings_ls, desc="extract embedding from batch..."):
    for embedding in batch:
        all_text_embeddings.append(embedding)

for batch in tqdm(labels_ls, desc="extract embedding from batch..."):
    for label in batch:
        all_labels.append(label)

# print(all_text_embeddings)
# print(all_labels)
print(len(all_text_embeddings))
print(len(all_labels))

# tensor to numpy
all_text_embeddings = [embedding.detach().numpy() for embedding in all_text_embeddings]
all_labels = [label.detach().numpy for label in all_labels]

df = pd.DataFrame(columns=["text_embedding", "label"])
df["text_embedding"] = all_text_embeddings
df["label"] = all_labels

# save to csv
df.to_csv("dataset/text_embedding.csv", sep="\t", encoding="utf-8", index=False)

# os.environ["TORCH"] = torch.__version__
# print(torch.__version__)
# # %pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# # %pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# # %pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
# # %pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
