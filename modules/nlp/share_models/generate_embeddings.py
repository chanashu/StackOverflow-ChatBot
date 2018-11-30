"""
train corpus contains similar sentences at the same row.
validation corpus contains the following columns: question, similar question, negative example 1, negative example 2, ...
test corpus contains the following columns: question, example 1, example 2, ...
"""

import os
from subprocess import Popen, PIPE
import sys
import numpy as np
import pandas as pd
from modules.nlp.share_models import utils

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(curr_dir_path, "models")

train_corpus = os.path.join(models_path, "train.tsv")
prepared_train_corpus = os.path.join(models_path, "prepared_corpus.tsv")
starspace_embedding_file = os.path.join(models_path, "word_embeddings.tsv")
starspace_bash_script = os.path.join(curr_dir_path, "starspace_train.sh")

starspace_embeddings = dict()
starspace_embedding_dimension = None


class GenerateStarSpaceEmbeddings:

    def __init__(self):
        pass

    def reload(self):
        try:
            print("Generating the prepared_train_corpus...")
            utils.prepare_file(train_corpus, prepared_train_corpus)
            print("Done...")
            print("Generating the StarSpace Embeddings using the prepared_train_corpus file...")

            # uncomment the below if you want to generate the embeddings
            # process, error = Popen(['bash', starspace_bash_script], stderr=PIPE, stdout=PIPE).communicate()
            # for line in process.stdout:
            #     sys.stdout.write(line)
            print("Done...")
        except Exception as e:
            print("Exception Occured....")

    def load(self):
        global starspace_embeddings
        global starspace_embedding_dimension
        print("updating the starspace embedding...")
        starspace_embeddings, starspace_embedding_dimension = utils.load_embeddings(starspace_embedding_file)
        print("Length of embeddings %s Embeddings Dimension is %s" %(len(starspace_embeddings),
                                                                     starspace_embedding_dimension))
        print("Done...")


if __name__ == "__main__":
    genObj = GenerateStarSpaceEmbeddings()
    print("Reload Start")
    #genObj.reload()
    print("Reload Done ...")
    genObj.load()
    print("Done...")