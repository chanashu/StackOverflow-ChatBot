import pandas as pd
from modules.nlp.share_models import utils
from modules.nlp.classifier.intent_classifier import stackoverflow_file
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(curr_dir_path, "models")


class RankingModel:
    def __init__(self):
        pass

    def reload(self):
        from modules.nlp.share_models.generate_embeddings import starspace_embeddings, starspace_embedding_dimension
        print("Ranking Model Reload Started...")
        try:
            posts_df = pd.read_csv(stackoverflow_file, sep='\t')
            counts_by_tag = posts_df.groupby(['tag']).agg({'tag': "count"})['tag'].to_dict()
            for tag, count in counts_by_tag.items():
                tag_posts = posts_df[posts_df['tag'] == tag]

                tag_post_ids = np.array([item for item in tag_posts["post_id"]])
                print(count)
                print(starspace_embedding_dimension)
                print(len(starspace_embeddings))
                tag_vectors = np.zeros((count,starspace_embedding_dimension), dtype=np.float32)
                for i, title in enumerate(tag_posts['title']):
                    tag_vectors[i, :] = utils.question_to_vec(title, starspace_embeddings,
                                                              starspace_embedding_dimension)
                print("tag %s shape %s" % (tag, tag_vectors.shape))

                # Dump post ids and vectors to a file.
                filename = os.path.join(models_path, os.path.normpath('%s.pkl' % tag))
                pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))
            print("Done...")

        except Exception as e:
            print("Exception Occurred while reloading the Ranking Model...")
            raise e


class ThreadRanker(object):
    # def __init__(self, paths):
    #     self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
    #     self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(models_path, tag_name + ".pkl")
        thread_ids, thread_embeddings = utils.unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        from modules.nlp.share_models.generate_embeddings import starspace_embeddings, starspace_embedding_dimension
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = utils.question_to_vec(question, starspace_embeddings, starspace_embedding_dimension)
        best_thread = pairwise_distances_argmin(thread_embeddings, question_vec.reshape(1, -1), axis=0,
                                                metric="cosine")
        return thread_ids[best_thread][0]
