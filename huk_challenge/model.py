import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from tqdm import tqdm
import emoji

nltk.download("stopwords")
nltk.download('punkt')


class SentimentPreprocessor:

    def __init__(self) -> None:
        glove_embedding = WordEmbeddings('glove')
        self.sentence_embedding = DocumentPoolEmbeddings([glove_embedding])
        self.target_mapping = {
            "Negative": -1,
            "Neutral": 0,
            "Positive": 1,
            "Irrelevant": 99
        }

    def preprocess_tweet(self, tweet) -> str:
        func_list = [
            self._normalize,
            self._naive_remove_links,
            self._replace_emoji,
            self._treat_twitter_entities,
            self._tokenize,
            self._token_level_operations
            ]

        for func in func_list:
            tweet = func(tweet)
        return tweet

    def preprocess_target(self, target):
        target = self.target_mapping[target]
        return target

    def preprocess(self, tweets, targets):
        x = []
        Y = []
        for tweet, target in tqdm(zip(tweets, targets)):
            target = self.preprocess_target(target)
            preprocessed_tweet = self.preprocess_tweet(tweet)
            if preprocessed_tweet != "" and preprocessed_tweet != " ":
                embedded_tweet = self._embed_tweet(preprocessed_tweet)
                x.append(embedded_tweet)
                Y.append(target)
        return x, Y

    def _token_level_operations(self, tweet):
        stops = stopwords.words("english")
        punctuation = string.punctuation
        punctuation = punctuation.replace("@", "")
        punctuation = punctuation + " " + "w/" + "`" + "’" + "‘"
        tweet = [token for token in tweet if token not in stops and token not in punctuation]
        tweet = " ".join(tweet)
        return tweet

    def _normalize(self, tweet):
        return tweet.lower()

    def _treat_twitter_entities(self, tweet):
        tweet = tweet.replace("@", "te")
        return tweet

    def _tokenize(self, tweet):
        tweet = word_tokenize(tweet)
        return tweet

    def _naive_remove_links(self, tweet):
        tweet = re.sub("[^ ]+\.[^ ]+", "", tweet)
        return tweet

    def _embed_tweet(self, tweet):
        sentence = Sentence(tweet)
        self.sentence_embedding.embed(sentence)
        return sentence.embedding.detach().cpu().numpy()

    def _replace_emoji(self, tweet):
        #tweet = emoji.replace_emoji(tweet, replace=lambda chars, _: " ".join(emoji.demojize(chars, delimiters=(" ", " ")).split("_")))
        tweet = emoji.demojize(tweet, delimiters=(" ", " "))
        return tweet


class SKSentimentModel:

    def __init__(self, model) -> None:
        self.preprocessor = SentimentPreprocessor()
        self.model = model

    def train(self, tweets, targets) -> None:
        x, Y = self.preprocessor.preprocess(tweets, targets)
        print("Preprocessing finished!")
        self.model.fit(x, Y)
        print("Model training finished!")

    def predict(self, tweet) -> str:
        preprocessed_tweet = self.preprocessor.preprocess_tweet(tweet)
        prediction = 0
        if preprocessed_tweet != "":
            embedded_tweet = self.preprocessor._embed_tweet(preprocessed_tweet).reshape(1, -1)
            prediction = self.model.predict(embedded_tweet)
            prediction = prediction.item()
        return prediction


if __name__ == "__main__":
    tweet = 'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣🤣🤣'
    preprocessor = SentimentPreprocessor()
    preprocessed_tweet = preprocessor.preprocess_tweet(tweet)
    embedded_tweet = preprocessor._embed_tweet(preprocessed_tweet)
    print(preprocessed_tweet)
