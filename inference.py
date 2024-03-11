import argparse
import torch

from models.n_gram import NGramModel
torch.manual_seed(313)

parser = argparse.ArgumentParser(description='Infer using a model')
parser.add_argument('--model_name', type=str, default='ngram', help="Name of the model to use for inference (default: %(default)s)")
parser.add_argument('--n_gram', type=int, default=2, help='Size of the n-gram (default: %(default)s)')
parser.add_argument('--x', type=str, default='', help='Input string (default: %(default)s)')
parser.add_argument('--max_new_tokens', type=str, default='', help='Maximum number of tokens to generate (default: %(default)s)')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model_name
    x = args.x
    max_new_tokens = args.max_new_tokens

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    if model_name == 'ngram':
        tokenizer = torch.load('model_checkpoints/{model_name}_tokenizer.pt')
        model = NGramModel(n_gram=args.n_gram, vocab_size=tokenizer.vocab_size, embedding_dim=256)
        model.load_state_dict(torch.load('model_checkpoints/{model_name}.pt'))
        model.eval()

    generated = model.generate(torch.tensor([tokenizer.encode('This is the best action')]).to(device), max_new_tokens)

    print(f'Input: {x}')
    print(f'Generated: {tokenizer.decode(generated[0])}')