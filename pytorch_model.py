import numpy as np
import torch
from torch import nn


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len), dtype=int)

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1

    ### END CODE HERE ###

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix))

    return embedding_layer


class emoji_model(nn.Module):
    def __init__(self, word_to_vec_map, word_to_index):
        super(emoji_model, self).__init__()
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        self.emb = embedding_layer
        self.LSTM = nn.LSTM(input_size=50, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=128, out_features=5)
        self.soft = nn.Softmax(0)

    def forward(self, x):
        x = self.emb(x)
        x, (h, c) = self.LSTM(x)
        x = self.drop(x[:, -1, :])
        x = self.linear(x)
        x = self.soft(x)
        return x
