import torch
from torch.utils.data import Dataset
import jieba
import os

from tqdm import tqdm


class IMDBDataset(Dataset):

    def __init__(self, data_dir, word2index, text_max_len=200):
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            file = [os.path.join(root, filename) for filename in files if not filename.startswith('.')]
            file_list.extend(file)

        self.x = []
        for file_name in tqdm(file_list, desc='Loading data: '):
            with open(file_name, 'r') as file:
                file_data = []
                for line in file.readlines():
                    if len(file_data) >= text_max_len:
                        break
                    line_words = jieba.cut(line.strip())
                    for word in line_words:
                        if len(file_data) >= text_max_len:
                            break
                        if word.lower() not in word2index.keys():
                            file_data.append(word2index['<oov>'])
                        else:
                            file_data.append(word2index[word.lower()])
                while len(file_data) < text_max_len:
                    file_data.append(word2index['<pad>'])
                self.x.append(file_data)

        self.y = []
        label_name_list = [file.split("/")[2] for file in file_list]
        for label_name in label_name_list:
            if label_name == "neg":
                self.y.append(0)
            elif label_name == "pos":
                self.y.append(1)

    def __getitem__(self, index):
        x_tensor = torch.tensor(self.x[index])
        y_tensor = torch.tensor(self.y[index])
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    dataset = IMDBDataset('aclImdb/train')
    print(dataset)