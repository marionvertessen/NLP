import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

from RNN import RNN

parameters = {
    "BATCH_SIZE": 64,
    "LR": 0.001,
    "EPOCHS": 10
}


"""
Cette fonction lit un fichier et retourne la liste des textes et la liste des émotions
"""
def load_file(file):
    phrases_split = []
    phrases = []
    emotions = []
    with open (file, "r") as f:
        lignes = f.readlines()
        for ligne in lignes:
            ligne = ligne.split(";")
            #On fait des listes de listes pour pouvoir utiliser le build vocab
            phrases_split.append(ligne[0].split())
            phrases.append(ligne[0])
            emotions.append(ligne[1].strip("\n").split())

    return phrases, phrases_split, emotions

def make_vocab (phrases):
    resultat = []
    for phrase in phrases:
        resultat.append(vocab(phrase))
    return resultat

def encode (onehot, line):
    return onehot(torch.tensor([line]), num_classes=15213)

def create_batch (batch_size, data_size, data):
    indexes = np.random.choise (data_size, size=batch_size, replace=False)
    return data[indexes]

if __name__ == '__main__':
    phrase_train, phrase_split_train, emotion_train = load_file("dataset/train.txt")
    phrase_test, phrase_split_test, emotion_test = load_file("dataset/test.txt")
    phrase_val, phrase_split_val, emotion_val = load_file("dataset/val.txt")

    # Extend text : they have to have the same dimension
    max_size_phrase = max(len(phrase) for phrase in phrase_split_train)
    for line in phrase_split_train:
        line.extend([''] * (max_size_phrase - len(line)))

    # Make the vocabulary lists for text and emotion
    vocab = build_vocab_from_iterator(phrase_split_train)
    train_voc = make_vocab(phrase_split_train)

    voc_emotion = build_vocab_from_iterator(emotion_train)
    emotion_voc = make_vocab(emotion_train)

    # We define the one hot encoder for text and feelings
    onehot = F.one_hot
    onehot_emotion = F.one_hot
    #print(encode(onehot, train_voc[0]))

    #one_hot_emotion = encode(onehot_emotion, emotion_voc[0], len(emotion_voc))

    input_size = len(vocab)
    hidden_size = 128
    output_size = len(voc_emotion)
    emb_size = 128

    # We define our model
    model = RNN(input_size, hidden_size, output_size, emb_size)

    #Define optimizer
    optimizer = Adam(model.parameters(), lr=parameters['LR'])

    #Define LOSS
    loss_function = nn.NLLLoss()

    # Define the hidden size
    hidden = torch.zeros(1, hidden_size)



    #for k in range(parameters['BATCH_SIZE']):
    print(train_voc[0][0])
    input = encode(onehot, phrase_split_train[0][0])
    print(input)

    output, next_hidden = model(input, hidden)
    print(output)




