

class SentimentPreprocessor:

    def __init__(self) -> None:
        pass

    def preprocess_tweet(self, tweet) -> str:
        pass

    def preprocess(self, dataset):
        pass


class SentimentModel:

    def __init__(self) -> None:
        self.preprocessor = SentimentPreprocessor()

    def train(self, dataset) -> None:
        pass

    def predict(self, tweet) -> str:
        pass


if __name__ == "__main__":
    pass
