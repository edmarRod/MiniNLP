# Language Models

This repository contains implementations of various language models using PyTorch. You can use these models for training and inference tasks.

## Quick Start

1. **Install Requirements**: First, ensure you have Poetry installed. Then, install the required dependencies using Poetry.

    ```sh
    poetry install
    ```

2. **Get the Data**: The code uses the IMDb reviews dataset. Run the following command to download and prepare the dataset:

    ```sh
    python datasets/get_imdb_dataset.py
    ```

3. **Training**: Start the training process by running `main.py`.

    ```sh
    python main.py
    ```

4. **Inference**: Use the model and tokenizer associated for inference by running `inference.py`.
    ```sh
    python inference.py
    ```

## Models

Currently, the following models are implemented:

- **N-gram Model**: A classic N-gram language model, which uses the last n tokens to predict the next one.
- **Continuous Bag of Words (CBOW)**: A neural network-based language model architecture. Used to generate word embeddings. [Reference Paper](https://arxiv.org/pdf/1301.3781.pdf)
- **Decoder-only Language Model**: A decoder-only architecture like those used in LLMs, based on the classic transformer model. [Reference Paper](https://arxiv.org/pdf/1706.03762.pdf)


## Examples

### Training
```sh
python main.py --model_name ngram --context_size 16 --batch_size 16 --max_epochs 10 --vocab_size 20000 --n_gram 2 --embedding_dim 256 --num_heads 8 --dim_feedforward 1024 --num_blocks 4 --save_model True --debug False
```

### Inference

```sh
python inference.py --model_name ngram --context_size 16 --vocab_size 20000 --n_gram 2 --embedding_dim 256 --num_heads 8 --dim_feedforward 1024 --num_blocks 4 --x 'Here we go again' --max_new_tokens 10
```
