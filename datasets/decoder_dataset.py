import torch
from torch.utils.data import Dataset

class DecoderDataset(Dataset):
  def __init__(self, data, context_size:int = 8):
    self.x_chunks = []
    self.y_chunks = []
    for line in data:
        if len(line) < context_size + 1:
            continue
        for i in range(len(line)-context_size):
            self.x_chunks.append(line[i:i+context_size])
            self.y_chunks.append(line[i+1:i+context_size+1])

  def __len__(self):
    return len(self.x_chunks)

  def __getitem__(self, idx):
    return torch.LongTensor(self.x_chunks[idx]), torch.LongTensor(self.y_chunks[idx])
  
if __name__ == '__main__':
    from tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    
    sample_data = ["This is some sample text for testing.", "This is another sample text."]
    tokenizer = WhitespaceTokenizer()
    tokenizer.create_vocab(''.join(sample_data))
    sample_data = [tokenizer.encode(line) for line in sample_data]
    context_size = 4
    dataset = DecoderDataset(data=sample_data, context_size=context_size)
    x, y = dataset[0]
    print([tokenizer.decode(aux) for aux in dataset.x_chunks])
    print([tokenizer.decode(aux) for aux in dataset.y_chunks])

    print(f'Original data: {tokenizer.decode(sample_data[0])}')
    print("X:", tokenizer.decode(x), x)
    print("Y:", tokenizer.decode(y), y)