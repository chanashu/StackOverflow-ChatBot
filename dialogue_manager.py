import os
from chatterbot import ChatBot
from modules.nlp.share_models import utils
from modules.nlp.ranking_model.ranking import ThreadRanker


class DialogueManager(object):
    def __init__(self):
        from modules.nlp.classifier.intent_classifier import intent_recognizer, tfidf_vectorizer, tag_classifier
        # # Intent recognition:
        self.intent_recognizer = intent_recognizer
        self.tfidf_vectorizer = tfidf_vectorizer

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'
        #
        # # Goal-oriented part:
        self.tag_classifier = tag_classifier
        self.thread_ranker = ThreadRanker()
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""
        self.chitchat_bot = ChatBot('Chandru Bot', trainer="chatterbot.trainers.ChatterBotCorpusTrainer")

        # Train the chatbot based on the english corpus
        self.chitchat_bot.train("chatterbot.corpus.english")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        prepared_question = utils.text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            response = self.chitchat_bot.get_response(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)

            return self.ANSWER_TEMPLATE % (tag, thread_id)