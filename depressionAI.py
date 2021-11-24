import numpy as np
import pandas as pd
import pickle
import os
import random

import spacy
from spacy.util import minibatch
from spacy.training.example import Example

data = pd.read_csv('../input/suicide-watch/Suicide_Detection.csv')
data.head(10)

nlp = spacy.blank("en")
categorizer = nlp.add_pipe("textcat")
categorizer.add_label("suicide")
categorizer.add_label("non-suicide")

trainX = data['text'].values
trainY = [{'cats': {'suicide': label == 'suicide', 'non-suicide': label == 'non-suicide'}} for label in data['class']]
trainData = list(zip(trainX, trainY))

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()
epochs = 10

losses = {}
for epoch in range(epochs):
    random.shuffle(trainData)
    batches = minibatch(trainData, size=8)
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)

message = input('- ')
docs = [nlp.tokenizer(message)]

categories = nlp.get_pipe('textcat')
scores = categories.predict(docs)[0]
prediction = scores.argmax()

print(f'Prediction: {categories.labels[prediction]} with a certainty of {int(max(scores)*100)}%')
