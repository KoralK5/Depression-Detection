import pickle
import spacy

f = open('model.pkl', 'rb')
nlpModel = pickle.load(f)
categories = nlpModel.get_pipe('textcat')

message = ' '
while message != '':
    message = input('- ')
    docs = [nlpModel.tokenizer(message)]

    scores = categories.predict(docs)[0]
    prediction = categories.labels[scores.argmax()]
    prediction = ('❌😞 bad' if prediction=='suicide' else '✔️😊 good') + ' mental health'

    print(f'\nPrediction: {prediction} with a certainty of {int(max(scores)*100)}%\n\n')
