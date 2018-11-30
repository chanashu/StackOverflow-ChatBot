import requests
import time
import argparse
import os
import json
from dialogue_manager import DialogueManager
from requests.compat import urljoin
from modules.nlp.share_models import utils


from modules.nlp.share_models.generate_embeddings import GenerateStarSpaceEmbeddings
from modules.nlp.classifier.intent_classifier import Classifier
from modules.nlp.ranking_model.ranking import RankingModel

generate_emb_obj = GenerateStarSpaceEmbeddings()
classfier_obj = Classifier()
rank_obj = RankingModel()


def load():
    print("Loading all the modules ....")
    try:
        generate_emb_obj.load()
    except  Exception as e:
        print("Exception Occurred while loading the Embeddings")
        raise e

    try:
        classfier_obj.load()
    except Exception as e:
        print("Exception Occurred while loading the classifier...")
        raise e

    # try:
    #     rank_obj.reload()
    # except Exception as e:
    #     print("Exception Occurred while loading the Ranking Model...")
    #     raise e
    print("Loaded all the Modules ...")


def reload():
    print("Reloading all the modules ....")
    try:
        generate_emb_obj.reload()
        generate_emb_obj.load()
    except Exception as e:
        print("Exception Occurred while Reloading the Embeddings")
        raise e

    try:
        classfier_obj.reload()
        classfier_obj.load()
    except Exception as e:
        print("Exception Occurred while Reloading the classifier...")
        raise e

    try:
        rank_obj.reload()
    except Exception as e:
        print("Exception Occurred while Reloading the Ranking Model...")
        raise e
    print("Reloading Completed...")


class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        print("API URL is %s" % (self.api_url))
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        print("INSIDE GET ANSWER is %s" % (question))
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        return self.dialogue_manager.generate_answer(question)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())


class SimpleDialogueManager(object):
    """
    This is the simplest dialogue manager to test the telegram bot.
    Your task is to create a more advanced one in dialogue_manager.py."
    """

    def generate_answer(self, question):
        return "Hello, world!"

def main():
    args = parse_args()
    token = args.token

    if not token:
        if not "TELEGRAM_TOKEN" in os.environ:
            print("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
            return
        token = os.environ["TELEGRAM_TOKEN"]

    #################################################################
    # triggers the reload of all the modules
    #reload()
    # triggers the load of all the modules
    load()
    #dialogue_manager = SimpleDialogueManager()
    dialogue_manager = DialogueManager()
    bot = BotHandler(token, dialogue_manager)

    ###############################################################

    print("Ready to talk!")
    offset = 0
    while True:
        print("INSIDE TRUE")
        updates = bot.get_updates(offset=offset)
        print("updates %s" % (updates))
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    print("text is %s" % (text))
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)


if __name__ == "__main__":
    main()
