import argparse
import math
from collections import defaultdict, Counter
import numpy as np

def preprocess_data(filename):
    """
    Input: Filename

    Return a tuple (dict, dict, list) where each dict maps names to  sentence matrices and sizes arrays (
    first is train, second is validation);the list is the vocabulary
    """
    token_counter = Counter()
    size_counter = Counter()
    max_size = 30
    valid_proportion = 0.2

    # first pass to build vocabulary and count sentence sizes
    print('Creating vocabulary...')
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            sent_size = len(tokens)
            if sent_size > max_size:
                continue

            # keep track of different size bins, with bins for
            # 1-10, 11-20, 21-30, etc
            top_bin = int(math.ceil(sent_size / 10) * 10)
            #size_counter is a counter which stores the bins and number of sentences in each bin
            size_counter[top_bin] += 1
            #roken_counter is a counter which stores tokens and its count
            token_counter.update(tokens)
    
    
    # sort it keeping the order
    vocabulary = [w for w, count in token_counter.most_common()]
    vocabulary.insert(0, '</s>')
    vocabulary.insert(1, '<unk>')
    #mapping will map the vocabulary to a sequence number
    mapping = zip(vocabulary, range(len(vocabulary)))
    dd = defaultdict(lambda: 1, mapping)
    print("vocabulary created...")

    print("creating sentence matrix...")
    train_data = {}  # dictionary to be used with numpy.savez
    valid_data = {}
    for max_sent_size in size_counter:
        if max_sent_size !=0:
            min_sent_size = max_sent_size-9
        else:
            min_sent_size = max_sent_size
        num_sentence = size_counter[max_sent_size]
        print('converting %d sentences which are between %d and %d' % (num_sentence, min_sent_size,max_sent_size))
        sents, sizes = create_sent_matrix(filename, num_sentence, min_sent_size, max_sent_size, dd)

        # shuffle sentences and sizes with the sime RNG state
        state = np.random.get_state()
        np.random.shuffle(sents)
        np.random.set_state(state)
        np.random.shuffle(sizes)

        ind = int(len(sents) * valid_proportion)
        valid_sentences = sents[:ind]
        valid_sizes = sizes[:ind]
        train_sentences = sents[ind:]
        train_sizes = sizes[ind:]

        train_data['sentences-%d' % max_sent_size] = train_sentences
        train_data['sizes-%d' % max_sent_size] = train_sizes
        valid_data['sentences-%d' % max_sent_size] = valid_sentences
        valid_data['sizes-%d' % max_sent_size] = valid_sizes


        print('Numeric representation ready')
    return train_data, valid_data, vocabulary

def create_sent_matrix(path, num_sentence, min_sent_size, max_sent_size, dd):
    """
    Create a sentence matrix from the file in the given path.
    :param path: path to text file
    :param min_size: minimum sentence length, inclusive
    :param max_size: maximum sentence length, inclusive
    :param num_sentences: number of sentences expected
    :param word_dict: mapping of words to indices

    :return: tuple (2-d matrix, 1-d array) with sentences and
        sizes
    """

    sent_matrix = np.full((num_sentence, max_sent_size), 0, np.int32)
    sizes = np.empty(num_sentence,np.int32)
    i = 0
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            tokens = line.split()
            sent_size = len(tokens)
            if sent_size < min_sent_size or sent_size > max_sent_size:
                continue

            array = np.array([dd[token] for token in tokens])
            sent_matrix[i, :sent_size] = array
            sizes[i] = sent_size
            i += 1
    
    return sent_matrix, sizes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file previously tokenized (by whitespace) and preprocessed')

    args = parser.parse_args()
    filename = args.input
    train_data, val_data, vocab = preprocess_data(filename)