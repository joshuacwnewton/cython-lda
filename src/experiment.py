from argparse import ArgumentParser
from corpus import *
from lda import *

from contextlib import contextmanager
from time import time


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start

    print(f"For '{description}', time elapsed: {ellapsed_time}\n")


def main():
    defaultTopicCount = 100
    defaultIterationCount = 1000

    # parse command-line arguments

    p = ArgumentParser()

    p.add_argument('-input_file', metavar='input-file', help='file containing preprocessed data')
    p.add_argument('-output_dir', metavar='output-dir', default='../output', help='output directory')
    p.add_argument('-T', metavar='num-topics', type=int, default=defaultTopicCount, help='number of topics (default: {})'.format(defaultTopicCount))
    p.add_argument('-S', metavar='num-iterations', type=int, default=defaultIterationCount, help='number of Gibbs sampling iterations (default: {})'.format(defaultIterationCount))
    p.add_argument('--optimize', action='store_true', help='optimize Dirichlet hyperparameters')

    args = p.parse_args()

    print(' args  ', args)

    corpus = Corpus.load(args.input_file)

    lda = LDA(corpus, args.T, args.S, args.optimize, args.output_dir)

    with timing("Python/NumPy"):
        lda.inference()
    with timing("Cython"):
        lda.cy_inference()


if __name__ == '__main__':
    main()
