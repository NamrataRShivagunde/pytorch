import argparse
import math
from collections import defaultdict, Counter

def preprocess_data(filename):
    token_counter = Counter()
    size_counter = Counter()
    max_size = 30

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
    print("vocabulary created...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file previously tokenized '
                                      '(by whitespace) and preprocessed')

    args = parser.parse_args()
    filename = args.input
    processed_file = preprocess_data(filename)