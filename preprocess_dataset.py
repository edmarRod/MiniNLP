# Downloads and unpacks the imdb reviews dataset, then joins all the reviews in a single text file and cleans up
import subprocess
import os
import glob

if not os.path.exists('data/final_text.txt'):
    if not os.path.exists('data'): 
        os.makedirs('data')
    subprocess.run('wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'.split())
    subprocess.run('tar -zxvf aclImdb_v1.tar.gz -C data'.split())


    with open('data/final_text.txt', 'w') as final_file:
        for file in glob.glob('data/aclImdb/*/*/*.txt'):
            with open(file, 'r') as cur_text_file:
                for line in cur_text_file:
                    final_file.write(line)
                    final_file.write('\n')


    subprocess.run('rm aclImdb_v1.tar.gz'.split())
    subprocess.run('rm -r data/aclImdb'.split())