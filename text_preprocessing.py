# Text preprocessing using Pipeline Design Pattern

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def process(self, text):
        for step in self.steps:
            text = step(text)
        return text

class Tokenization:
    def __call__(self, text):
        return word_tokenize(text)

class StopWordRemoval:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def __call__(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

class Stemming:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

class Lemmatization:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

class Word2VecEmbedding:
    def __init__(self, model_file):
        self.model = Word2Vec.load(model_file)

    def __call__(self, tokens):
        return [self.model[word] for word in tokens if word in self.model.wv.vocab]

class TFIDFVectorization:
    def __init__(self, max_features=None):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def __call__(self, tokens):
        return self.vectorizer.transform(tokens)

# Create a pipeline
pipeline = TextPreprocessingPipeline()

# Add steps to the pipeline
pipeline.add_step(Tokenization())
pipeline.add_step(StopWordRemoval())
pipeline.add_step(Stemming())
pipeline.add_step(Lemmatization())
pipeline.add_step(Word2VecEmbedding("word2vec_model.bin"))
pipeline.add_step(TFIDFVectorization())

# Process a text
text = "This is a sample text for NLP preprocessing."
processed_text = pipeline.process([text])

print(processed_text)