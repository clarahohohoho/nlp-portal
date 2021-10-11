import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax

def sen_spacy(text):

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(text)

    if doc._.polarity < 0:
        return 'Negative', doc._.polarity, 'Polarity: '

    elif doc._.polarity == 0:
        return 'Neutral', doc._.polarity, 'Polarity: '

    elif doc._.polarity > 0:
        return 'Positive', doc._.polarity, 'Polarity: '

def sen_hf(text):

    mapping = {0:'Negative', 1:'Neutral', 2:'Positive'}

    model = AutoModelForSequenceClassification.from_pretrained("./modules/sentiment/twitter-roberta-base-sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./modules/sentiment/twitter-roberta-base-sentiment_tokenizer")

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    maxarg = np.argmax(scores)
    for key in mapping.keys():
        if maxarg == key:
            sentiment = mapping[key]

    return sentiment, max(scores), 'Probability: '

def main_sen(text, model):

    if model == 'spacy':
        sentiment, score, score_type = sen_spacy(text)

    elif model == 'huggingface':
        sentiment, score, score_type = sen_hf(text)

    return sentiment, score, score_type