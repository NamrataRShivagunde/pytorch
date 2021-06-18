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
            size_counter[top_bin] += 1
            token_counter.update(tokens)
    
    print(top_bin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Text file previously tokenized '
                                      '(by whitespace) and preprocessed')

    args = parser.parse_args()
    filename = args.input
    processed_file = preprocess_data(filename)