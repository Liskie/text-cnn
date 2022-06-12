import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self,
                 embedding_size=100,
                 hidden_size=256,
                 vocab_size=10000,
                 text_max_len=40,
                 label_num=2):
        super(TextCNN, self).__init__()
        self.model_name = str(type(self))
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.text_max_len = text_max_len
        self.label_num = label_num

        self.kernel_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.conv3 = nn.Conv1d(in_channels=embedding_size, out_channels=hidden_size, kernel_size=3)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_size, out_channels=hidden_size, kernel_size=kernel_size)
            for kernel_size in self.kernel_sizes
        ])
        self.poolings = nn.ModuleList([
            nn.MaxPool1d(kernel_size=self.text_max_len - kernel_size + 1)
            for kernel_size in self.kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(in_features=hidden_size * len(self.kernel_sizes), out_features=self.label_num)

    def forward(self, x):
        batch_size = x.shape[0]
        x_embed = self.embedding(x).permute(0, 2, 1)

        x_convs = [
            conv(x_embed)
            # self.relu(conv(x_embed))
            for conv in self.convs
        ]

        x_poolings = [
            pooling(x_conv)
            for x_conv, pooling in zip(x_convs, self.poolings)
        ]

        x_cat = torch.cat(x_poolings, -1)
        x_cat = x_cat.view(batch_size, -1)

        x_res = self.linear(x_cat)
        x_res = self.softmax(x_res)

        return x_res

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
