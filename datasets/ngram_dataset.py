import torch
from torch.utils.data import Dataset
from tokenizers.whitespace_tokenizer import WhitespaceTokenizer

class NgramDataset(Dataset):
  def __init__(self, data, n_gram: int):
    self.n_gram = n_gram
    self.x_chunks = []
    self.y_chunks = []
    for line in data:
        if len(line) < self.n_gram + 1:
            continue
        for i in range(len(line)-self.n_gram):
            self.x_chunks.append(line[i:i+self.n_gram])
            self.y_chunks.append(line[i+self.n_gram])

  def __len__(self):
    return len(self.x_chunks)

  def __getitem__(self, idx):
    return torch.LongTensor(self.x_chunks[idx]), torch.tensor(self.y_chunks[idx], dtype=torch.long)
  
if __name__ == '__main__':
    sample_data = ["This is some sample text for testing.", "This is another sample text."]
    tokenizer = WhitespaceTokenizer()
    tokenizer.create_vocab(''.join(sample_data))
    sample_data = [tokenizer.encode(line) for line in sample_data]
    n_gram = 2
    dataset = NgramDataset(data=sample_data, n_gram=n_gram)
    x, y = dataset[0]
    print([tokenizer.decode(aux) for aux in dataset.x_chunks])
    print([tokenizer.decode([aux]) for aux in dataset.y_chunks])

    print(f'Original data: {tokenizer.decode(sample_data[0])}')
    print("X:", tokenizer.decode(x), x)
    print("Y:", tokenizer.decode(y.unsqueeze(0)), y)