import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from sklearn.metrics import classification_report

from dataset import IMDBDataset
from model import TextCNN

torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--epoch_num', type=int, default=20)
parser.add_argument('--text_max_len', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, choices=['small', 'full'], default='small')
args = parser.parse_args()


if __name__ == '__main__':

    # Hyper params
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    epoch_num = args.epoch_num
    text_max_len = args.text_max_len
    learning_rate = args.lr
    batch_size = args.batch_size
    label_num = 2

    if args.dataset == 'small':
        train_data_dir = 'aclImdb/small-train'
        test_data_dir = 'aclImdb/small-train'
    elif args.dataset == 'full':
        train_data_dir = 'aclImdb/train'
        test_data_dir = 'aclImdb/test'
    else:
        raise NotImplementedError

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Init
    word2index = {
        '<pad>': 0,
        '<oov>': 1
    }
    with open('aclImdb/imdb.vocab', 'r') as vocab_file:
        for line in vocab_file:
            word2index[line.strip()] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}

    train_set = IMDBDataset(data_dir=train_data_dir,
                            word2index=word2index)

    train_iter = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=True)

    test_set = IMDBDataset(data_dir=test_data_dir,
                           word2index=word2index)

    test_iter = DataLoader(dataset=train_set,
                           batch_size=batch_size,
                           num_workers=2)

    model = TextCNN(embedding_size=embedding_size,
                    vocab_size=len(word2index),
                    hidden_size=hidden_size,
                    text_max_len=text_max_len,
                    label_num=label_num).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train
    losses = []
    for epoch in trange(epoch_num, desc='Training: '):
        for data, labels in train_iter:
            data = data.to(device)
            labels = labels.to(device)

            out = model(Variable(data))
            loss = criterion(out, Variable(labels))

            losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, current loss = {np.mean(losses):.06f}')
        losses = []
        # save the model in every epoch
        model.save(f'checkpoints/epoch_{epoch}.ckpt')

    # Test
    y_pred = []
    y_true = []

    for data, labels in tqdm(test_iter, desc='Testing: '):
        data = data.to(device)
        labels = labels.to(device)
        y_pred.extend(torch.argmax(model(Variable(data)), dim=-1).tolist())
        y_true.extend(labels.tolist())

    with open('out/result.txt', 'w') as out:
        for true, pred in zip(y_true, y_pred):
            out.write(f'{true} {pred}\n')

    print(classification_report(y_true, y_pred))
