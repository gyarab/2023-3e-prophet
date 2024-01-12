import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pandas as pd




#prepare sequence
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the "dataset" folder relative to the script
dataset_folder = os.path.join(script_dir, '..', 'dataset')

    # Construct the path to the "data" folder within the "dataset" folder
data_folder = os.path.join(dataset_folder, 'data')

    # Construct the full path to the CSV file
    #csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')
csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')
dataset_folder = os.path.join(script_dir, '..', 'dataset')

    # Construct the path to the "data" folder within the "dataset" folder
data_folder = os.path.join(dataset_folder, 'data')

    # Construct the full path to the CSV file
    #csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')
csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')

# Load data from CSV file
csv_file_path = os.path.join(data_folder, '60BTCUSDT.csv')
df = pd.read_csv(csv_file_path)

# Extract relevant columns (timestamp, close, low, high)
timestamp_sequence = df['timestamp'].astype(float).tolist()
close_prices_sequence = df['close'].astype(float).tolist()
low_prices_sequence = df['low'].astype(float).tolist()
high_prices_sequence = df['high'].astype(float).tolist()

# Create word-to-index dictionaries for each sequence
timestamp_to_ix = {word: idx for idx, word in enumerate(timestamp_sequence)}
close_prices_to_ix = {word: idx for idx, word in enumerate(close_prices_sequence)}
low_prices_to_ix = {word: idx for idx, word in enumerate(low_prices_sequence)}
high_prices_to_ix = {word: idx for idx, word in enumerate(high_prices_sequence)}

# Example usage:
# Replace the following lines with the appropriate sequences and dictionaries
# For example, use timestamp_to_ix for timestamp_sequence
# Use close_prices_to_ix for close_prices_sequence, and so on.
# Replace the training_data with your sequences and numerical values
training_data = [
    (timestamp_sequence, close_prices_sequence, low_prices_sequence, high_prices_sequence),
    # Add more tuples as needed
]




#create model
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    

#! ---------------------------------------------- Needs to be update to work with out values instead of words --------------------------------------------------------------
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)