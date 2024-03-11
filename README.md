# Language Models

Repo with the code to train and use different language models.

## Quick start

Install the requirements using poetry.

```sh
poetry install
```

Get the data used, which is the imdb reviews dataset.

```sh
python datasets/get_imdb_dataset.py
```

Run the training from main.py

```sh
python main.py
```

## Models

Currently implemented are N-gram, CBOW and a Decoder language models.