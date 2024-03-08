# Downloads and unpacks the imdb reviews dataset, then joins all the reviews, separates into train and validation sets and cleans up
import subprocess
import os
import glob
import random

if not os.path.exists('data/imdb_train.txt') and not os.path.exists('data/imdb_test.txt'):
    random.seed(42)
    if not os.path.exists('data'): 
        os.makedirs('data')
    subprocess.run('wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'.split())
    subprocess.run('tar -zxvf aclImdb_v1.tar.gz -C data'.split())

    #reads all reviews from file and randomly assigns to train or validation

    with open('data/imdb_train.txt', 'w') as imdb_train:
        with open('data/imdb_test.txt', 'w') as imdb_test:
            for file in glob.glob('data/aclImdb/*/*/*.txt'):
                with open(file, 'r') as cur_text_file:
                    if random.random() < 0.8:
                        for line in cur_text_file:
                            imdb_train.write(line)
                            imdb_train.write('\n')
                    else:
                        for line in cur_text_file:
                            imdb_test.write(line)
                            imdb_test.write('\n')



    subprocess.run('rm aclImdb_v1.tar.gz'.split())
    subprocess.run('rm -r data/aclImdb'.split())