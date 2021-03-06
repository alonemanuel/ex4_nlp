import os
import pickle
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data_loader

DEFAULT_LOGLINEAR_WEIGHT_DECAY = 0.0001

DEFAULT_LOGLINEAR_LR = 0.01

DEFAULT_LOGLINEAR_N_EPOCHS = 20

DEFAULT_LSTM_DROPOUT = 0.5

DEFAULT_LSTM_N_LAYERS = 1

DEFAULT_LSTM_HIDDEN_DIM = 100

W2V_EMBEDDING_DIM = 300

DEFAULT_BATCH_SIZE = 64

DEFAULT_LSTM_LR = 0.001

DEFAULT_LSTM_N_EPOCHS = 4

DEFAULT_LSTM_WEIGHT_DECAY = 0.0001
plt.style.use('seaborn-whitegrid')

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 10
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_average = np.zeros(embedding_dim, dtype=np.float)
    n_known_words = 0
    for word in sent.text:
        if word in word_to_vec:
            n_known_words += 1
            w2v_average += word_to_vec[word]
    if n_known_words != 0:
        w2v_average = w2v_average / n_known_words
    return w2v_average


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size, dtype=np.float)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    n_words = len(word_to_ind)
    sent_len = len(sent.text)
    average = np.zeros(n_words, dtype=np.float)
    for word in sent.text:
        average += get_one_hot(n_words, word_to_ind[word])
    average = average / sent_len
    return average


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: idx for idx, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    sent_embedding = np.zeros((seq_len, embedding_dim), dtype=np.float)
    for i, word in enumerate(sent.text):
        if i < seq_len and word in word_to_vec:
            sent_embedding[i, :] = word_to_vec[word]
    return sent_embedding


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        batch_size = text.size(0)
        h0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(get_available_device())
        c0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(get_available_device())

        lstm_out, (h_n, c_n) = self.lstm(text.float(), (h0, c0))
        out = self.fc(lstm_out[:, -1, :])

        return out

    def predict(self, text):
        return (self.forward(text) >= 0.5).float()
        # return torch.round(self.forward(text)).float()


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x.float())

    def predict(self, x):
        return (self.forward(x) >= 0.5).float()

        # return torch.round(self.forward(x)).float()


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return ((preds == y).sum() / len(preds)).item()


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    n_iters = 0
    epoch_loss = 0
    epoch_acc = 0
    device = get_available_device()
    criterion = criterion.to(device)
    for x_batch, y_batch in data_iterator:
        n_iters += 1
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        loss = criterion(model(x_batch), y_batch.unsqueeze(1))
        acc = binary_accuracy(model.predict(x_batch), y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / n_iters, epoch_acc / n_iters


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    n_iters = 0
    total_loss = 0
    total_acc = 0
    device = get_available_device()
    criterion = criterion.to(device)
    for x_batch, y_batch in data_iterator:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        n_iters += 1
        total_loss += criterion(model(x_batch), y_batch.unsqueeze(1))
        total_acc += binary_accuracy(model.predict(x_batch), y_batch.unsqueeze(1))
    return total_loss / n_iters, total_acc / n_iters


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    total_pred = torch.Tensor().to(get_available_device())
    for x_batch, y_batch in data_iter:
        x_batch, y_batch = x_batch.to(get_available_device()), y_batch.to(get_available_device())
        y_pred = model.predict(x_batch)
        total_pred = torch.cat((total_pred, y_pred))
    return total_pred.cpu().detach().numpy()


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    for epoch in range(n_epochs):
        model.train()
        train_loss, train_acc = train_epoch(model, data_manager.get_torch_iterator(), optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, data_manager.get_torch_iterator(VAL), criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        print(f'\nEpoch {epoch + 0:03}: | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f}',end='')
        print(f' | Valid Loss: {valid_loss:.5f} | Valid Acc: {valid_acc:.3f}',end='')
    return train_losses, train_accs, valid_losses, valid_accs


def get_test_acc(model, data_manager, subset=None):
    test_pred = get_predictions_for_data(model, data_manager.get_torch_iterator(TEST))
    test_labels = data_manager.get_labels(TEST)
    if subset == 'NEGATED':
        negated_subset_indices = data_loader.get_negated_polarity_examples(data_manager.sentences[TEST])
        test_pred = np.array((itemgetter(*negated_subset_indices)(test_pred)))
        test_labels = np.array((itemgetter(*negated_subset_indices)(test_labels)))
    elif subset == 'RARE':
        rare_subset_indices = data_loader.get_rare_words_examples(data_manager.sentences[TEST],
                                                                  data_manager.sentiment_dataset)
        test_pred = np.array((itemgetter(*rare_subset_indices)(test_pred)))
        test_labels = np.array((itemgetter(*rare_subset_indices)(test_labels)))
    test_acc = binary_accuracy(test_pred.flatten(), test_labels.flatten())
    return test_acc


def print_test_accuracies(model, data_manager):
    test_acc = get_test_acc(model, data_manager)
    negated_acc = get_test_acc(model, data_manager, 'NEGATED')
    rare_acc = get_test_acc(model, data_manager, 'RARE')
    print('test accuracy: {}\nnegated accuracy: {}\nrare accuracy: {}'.format(test_acc, negated_acc, rare_acc))


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    print('training log linear with 1-hot')
    data_manager = DataManager(batch_size=DEFAULT_BATCH_SIZE)
    model = LogLinear(data_manager.get_input_shape()[-1]).to(get_available_device())
    train_losses, train_accs, valid_losses, valid_accs = train_model(model, data_manager, 20, 0.01, weight_decay=0.0001)
    plot(train_losses, train_accs, valid_losses, valid_accs, 'Log Linear: 1-hot')
    print_test_accuracies(model, data_manager)
    return


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    print('training log linear with w2v')
    data_manager = DataManager(data_type=W2V_AVERAGE,
                               batch_size=DEFAULT_BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(data_manager.get_input_shape()[-1]).to(get_available_device())
    train_losses, train_accs, valid_losses, valid_accs = train_model(model=model,
                                                                     data_manager=data_manager,
                                                                     n_epochs=DEFAULT_LOGLINEAR_N_EPOCHS,
                                                                     lr=DEFAULT_LOGLINEAR_LR,
                                                                     weight_decay=DEFAULT_LOGLINEAR_WEIGHT_DECAY)
    plot(train_losses, train_accs, valid_losses, valid_accs, 'Log Linear: w2v')
    print_test_accuracies(model, data_manager)
    return


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    print('training LSTM with w2v')
    data_manager = DataManager(data_type=W2V_SEQUENCE,
                               batch_size=DEFAULT_BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=data_manager.get_input_shape()[-1],
                 hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
                 n_layers=DEFAULT_LSTM_N_LAYERS,
                 dropout=DEFAULT_LSTM_DROPOUT).to(get_available_device())
    train_losses, train_accs, valid_losses, valid_accs = train_model(model=model,
                                                                     data_manager=data_manager,
                                                                     n_epochs=DEFAULT_LSTM_N_EPOCHS,
                                                                     lr=DEFAULT_LSTM_LR,
                                                                     weight_decay=DEFAULT_LSTM_WEIGHT_DECAY)
    plot(train_losses, train_accs, valid_losses, valid_accs, 'LSTM')
    print_test_accuracies(model, data_manager)
    return


def plot(train_losses, train_accs, valid_losses, valid_accs, title):
    f, (ax1, ax2) = plt.subplots(1, 2)
    x = [i for i in range(len(train_losses))]
    ax1.plot(x, train_losses, label='Train')
    ax1.plot(x, valid_losses, label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.set_ylim([0, 1])

    ax2.plot(x, train_accs, label='Train')
    ax2.plot(x, valid_accs, label='Validation')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.set_ylim([0, 1])

    f.suptitle(title)

    plt.show()


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
