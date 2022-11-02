# wget https://raw.githubusercontent.com/sharsi1/russkiwlst/master/stat_russkiwlst_top_1M.txt
# pip install transformers
# pip install tokenizers

from itertools import islice
from random import shuffle

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch import nn


device = "cuda:0"


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(tokenizer.get_vocab_size(), 32)
        self.linear = nn.Linear(32, 1)

    def embed(self, x):
        x = self.embedding(x)
        x = nn.ReLU()(x)
        x = torch.mean(x, axis=1).squeeze(axis=-1)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        return x


def prepare_tokenizer_training_dataset():
    with open("stat_russkiwlst_top_1M.txt") as f:
        lines = [x.strip().strip("\n") for x in f.readlines() if len(x.split()) == 2]
        dataset = []
        for x in lines:
            frequency, password = x.split()
            for _ in range(int(frequency)):
                dataset.append(password)

    with open("tokenizer_train.txt", "w") as f:
        for line in dataset:
            f.write(f"{line}\n")


def train_tokenizer():
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>"],
                         vocab_size=20000,
                         min_frequency=2)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.train(files=["tokenizer_train.txt"], trainer=trainer)
    return tokenizer


def prepare_classifier_training_dataset():
    dataset = []

    with open("stat_russkiwlst_top_1M.txt") as f:
        lines = [x.strip().strip("\n") for x in f.readlines() if len(x.split()) == 2]
        for n, x in enumerate(lines):
            frequency, password = x.split()
            dataset.append((1.0, password))

    with open("negative_samples.txt") as f:
        lines = [x.strip().strip("\n") for x in f.readlines()]
        for line in lines:
            dataset.append((0, line))

    shuffle(dataset)
    train, test = dataset[:-10000], dataset[-10000:]

    return train, test


def training_loader(dataset, step, tokenizer):
    for i in range(0, len(dataset) - step, step):
        frequencies, passwords = zip(*dataset[i:i + step])
        passwords = [tokenizer.encode(x).ids for x in passwords]
        max_len = max(len(x) for x in passwords)
        for i in range(len(passwords)):
            current = passwords[i]
            current = current + [0] * (max_len - len(current))
            passwords[i] = current

        yield torch.tensor(passwords).to(device), torch.tensor(frequencies).to(device)


def train_model(train, test, model, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(50):
        model.eval()
        losses = []
        for x_batched, y_batched in training_loader(test, 200, tokenizer):
            with torch.no_grad():
                y_pred = model(x_batched)
                loss = criterion(y_pred.squeeze(-1), y_batched)
                losses.append(loss.cpu().item())
        print(f"EPOCH {epoch}, loss: {np.mean(losses)}")

        # put model in training mode
        model.train()

        for x_batched, y_batched in training_loader(train, 1000, tokenizer):
            y_pred = model(x_batched)
            loss = criterion(y_pred.squeeze(-1), y_batched)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def make_prediction(model, tokenizer, passwords):
    passwords = [tokenizer.encode(x).ids for x in passwords]
    max_len = max(len(x) for x in passwords)
    for i in range(len(passwords)):
        current = passwords[i]
        current = current + [0] * (max_len - len(current))
        passwords[i] = current
    with torch.no_grad():
        y_pred = model(torch.tensor(passwords).to(device))
    return y_pred.squeeze(1).cpu().numpy().tolist()


def compute_password_scores(model, tokenizer):
    with open("result_refined.txt") as f:
        while True:
            passwords = list(islice(f, 100000))
            predictions = make_prediction(model, tokenizer, passwords)
            with open("scored_passwords.txt", "a") as g:
                for prediction, password in zip(predictions, passwords):
                    if prediction > 0 and "\t" not in password:
                        g.write(f"{prediction}\t{password}\n")

            if not passwords:
                break


if __name__ == '__main__':
    prepare_tokenizer_training_dataset()
    tokenizer = train_tokenizer()
    model = Classifier().to(device)
    train, test = prepare_classifier_training_dataset()
    trained_model = train_model(train, test, model, tokenizer)
    compute_password_scores(trained_model, tokenizer)
