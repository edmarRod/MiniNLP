from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from datasets import NgramDataset
from datasets.cbow_dataset import CBOWDataset
from models.cbow import CBOW
from models.decoder_llm import MiniLLM
from models.n_gram import NGramModel
from tokenizers.whitespace_tokenizer import WhitespaceTokenizer
import argparse

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model_name', type=str, default='ngram', help="Name of the model to train (default: %(default)s)")
parser.add_argument('--context_size', type=int, default=16,
                    help='Size of the context (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size (default: %(default)s)')
parser.add_argument('--max_epochs', type=int, default=10,
                    help='Maximum number of epochs to train (default: %(default)s)')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='Vocabulary size for tokenizer (default: %(default)s)')
parser.add_argument('--n_gram', type=int, default=2,
                    help='Size of the n-gram (default: %(default)s)')
parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension (default: %(default)s)')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for attention (default: %(default)s)')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of transformer feedforward (default: %(default)s)')
parser.add_argument('--num_blocks', type=int, default=4, help='Number of decoder blocks (default: %(default)s)')
parser.add_argument('--save_model', type=bool, default=True,
                    help='Save the trained model (default: %(default)s)')
parser.add_argument('--debug', type=bool, default=False,
                    help='Print debug information')
args = parser.parse_args()

if __name__ == '__main__':
    context_size = args.context_size
    batch_size = args.batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    debug_mode = args.debug
    max_epochs = args.max_epochs
    n_gram = args.n_gram
    save_model = args.save_model
    vocab_size = args.vocab_size
    model_name = args.model_name
    embedding_dim = args.embedding_dim
    num_heads = args.num_heads
    dim_feedforward = args.dim_feedforward
    num_blocks=args.num_blocks

    torch.manual_seed(313)
    tokenizer = WhitespaceTokenizer(vocab_size=vocab_size)

    def read_dataset(path) -> list[str]:
        with open(path) as f:
            dataset = []
            for line in f:
                dataset.append(line+' <eos>')
                if debug_mode:
                    if len(dataset) > 100:
                        break

        return dataset

    train_data = read_dataset('data/imdb_train.txt')
    val_data = read_dataset('data/imdb_test.txt')

    tokenizer.create_vocab(''.join(train_data))

    train_data = [tokenizer.encode(line) for line in train_data]
    val_data = [tokenizer.encode(line) for line in val_data]



    if model_name == 'ngram':
        model = NGramModel(n_gram=n_gram, vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)
        train_dataloader = DataLoader(NgramDataset(data=train_data, n_gram=n_gram), batch_size=batch_size)
        val_dataloader = DataLoader(NgramDataset(data=val_data, n_gram=n_gram), batch_size=batch_size)
    elif model_name == 'cbow':
        model = CBOW(n_gram=n_gram, vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)
        train_dataloader = DataLoader(CBOWDataset(data=train_data, context_size=context_size), batch_size=batch_size)
        val_dataloader = DataLoader(CBOWDataset(data=val_data, context_size=context_size), batch_size=batch_size)
    elif model_name == 'mini_llm':
        model = MiniLLM(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, context_size = context_size, num_heads=num_heads, dim_feedforward=dim_feedforward, num_blocks=num_blocks)
        train_dataloader = DataLoader(NgramDataset(data=train_data, n_gram=n_gram), batch_size=batch_size)
        val_dataloader = DataLoader(NgramDataset(data=val_data, n_gram=n_gram), batch_size=batch_size)
    else:
        raise ValueError(f'Unknown model {model_name}')
    model.to(device=device)

    optimizer = torch.optim.AdamW(params = model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_dataloader, desc="Training epoch {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc="Validating epoch {}".format(epoch)):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
                val_loss += loss.item()
        print(f'Epoch {epoch}: Training loss: {train_loss / len(train_dataloader)}; Validation loss: {val_loss / len(val_dataloader)}')

    if save_model:
        torch.save(model.state_dict(), f'model_checkpoints/{model_name}.pt')
        torch.save(tokenizer, f'model_checkpoints/{model_name}_tokenizer.pt')

    if model_name != 'cbow':
        x = 'This is the best action'
        sentence = torch.tensor([tokenizer.encode(x)]).to(device)
        y = model.generate(sentence, 10)
        print(f"Prompt: {x}\nGenerated: {tokenizer.decode(y[0])}")