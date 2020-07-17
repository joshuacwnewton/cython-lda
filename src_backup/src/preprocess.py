import argparse, csv, re
from corpus import *
from numpy import *

from sklearn.datasets import fetch_20newsgroups


def create_stopword_list(f):

    if not f:
        return set()

    if isinstance(f, str):
        f = open(f, "r")

    return set(word.strip() for word in f)


def tokenize(data, stopwords=set()):

    tokens = re.findall('[a-z]+', data.lower())

    return [x for x in tokens if x not in stopwords]


def main():
    # parse command-line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', metavar='input-file', help='CSV file to be preprocessed')
    parser.add_argument('--remove-stopwords', metavar='stopword-file', help='remove stopwords provided in the specified file')
    parser.add_argument('--output-file', metavar='output-file', help='save preprocessed data to the specified file')

    args = parser.parse_args()

    # create stopword list

    stopwords = create_stopword_list(args.remove_stopwords)

    # preprocess data

    corpus = Corpus()

    if args.input_file == 'newsgroups':
        posts = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
                                   shuffle=True,
                                   random_state=1,
                                   return_X_y=True)
        N_SAMPLES = 2000
        subset = (posts[0][:N_SAMPLES], posts[1][:N_SAMPLES])
        for data, name in zip(subset[0], subset[1]):
            corpus.add(str(name), tokenize(data, stopwords))
    else:
        for name, _, data in csv.reader(open(args.input_file), delimiter='\t'):
            corpus.add(name, tokenize(data, stopwords))

    print('# documents =', len(corpus))
    print('# tokens =', sum(list(map(len, corpus))))
    print('# unique types =', len(corpus.alphabet))

    if args.output_file:
        corpus.save(args.output_file)


if __name__ == '__main__':
    main()
