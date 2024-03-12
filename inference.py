import argparse
import torch
from models.cbow import CBOW
from models.decoder_llm import MiniLLM

from models.n_gram import NGramModel
torch.manual_seed(313)

parser = argparse.ArgumentParser(description='Infer using a model')
parser.add_argument('--model_name', type=str, default='ngram', help="Name of the model to train (default: %(default)s)")
parser.add_argument('--context_size', type=int, default=16,
                    help='Size of the context (default: %(default)s)')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='Vocabulary size for tokenizer (default: %(default)s)')
parser.add_argument('--n_gram', type=int, default=2,
                    help='Size of the n-gram (default: %(default)s)')
parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension (default: %(default)s)')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for attention (default: %(default)s)')
parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of transformer feedforward (default: %(default)s)')
parser.add_argument('--num_blocks', type=int, default=4, help='Number of decoder blocks (default: %(default)s)')
parser.add_argument('--x', type=str, default='', help='Input string (default: %(default)s)', required=True)
parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of tokens to generate (default: %(default)s)')
args = parser.parse_args()

if __name__ == '__main__':
    context_size = args.context_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    n_gram = args.n_gram
    vocab_size = args.vocab_size
    model_name = args.model_name
    embedding_dim = args.embedding_dim
    num_heads = args.num_heads
    dim_feedforward = args.dim_feedforward
    num_blocks=args.num_blocks

    x = args.x
    max_new_tokens=args.max_new_tokens

    tokenizer = torch.load(f'model_checkpoints/{model_name}_tokenizer.pt')

    if model_name == 'ngram':
        model = NGramModel(n_gram=args.n_gram, vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'cbow':
        model = CBOW(n_gram=args.n_gram, vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim)
    elif model_name == 'mini_llm':
        model = MiniLLM(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, context_size = context_size, num_heads=num_heads, dim_feedforward=dim_feedforward, num_blocks=num_blocks)

    model.load_state_dict(torch.load(f'model_checkpoints/{model_name}.pt'))
    model.to(device)
    model.eval()

    generated = model.generate(torch.tensor([tokenizer.encode(x)]).to(device), max_new_tokens)

    print(f'Input: {x}')
    print(f'Generated: {tokenizer.decode(generated[0])}')