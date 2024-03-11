import torch
from torch.utils.data import Dataset

class CBOWDataset(Dataset):
  def __init__(self, data, context_size: int):
    self.context_size = context_size//2
    self.x_chunks = []
    self.y_chunks = []
    for line in data:
        if len(line) < self.context_size*2 + 1:
            continue
        for i in range(self.context_size, len(line)-self.context_size):
            before = line[i-self.context_size:i]
            after = line[i+1:i+self.context_size+1]
            before.extend(after)
            self.x_chunks.append(before)
            self.y_chunks.append(line[i])

  def __len__(self):
    return len(self.x_chunks)

  def __getitem__(self, idx):
    return torch.LongTensor(self.x_chunks[idx]), torch.tensor(self.y_chunks[idx], dtype=torch.long)
  
if __name__ == '__main__':
    from tokenizers.whitespace_tokenizer import WhitespaceTokenizer

    sample_data = ["This is some sample text for testing.", "This is another sample text."]
    tokenizer = WhitespaceTokenizer()
    tokenizer.create_vocab(''.join(sample_data))
    sample_data = [tokenizer.encode(line) for line in sample_data]
    context_size = 4
    dataset = CBOWDataset(data=sample_data, context_size=context_size)
    x, y = dataset[0]
    print([tokenizer.decode(aux) for aux in dataset.x_chunks])
    print([tokenizer.decode([aux]) for aux in dataset.y_chunks])

    print(f'Original data: {tokenizer.decode(sample_data[0])}')
    print("X:", tokenizer.decode(x), x)
    print("Y:", tokenizer.decode(y.unsqueeze(0)), y)