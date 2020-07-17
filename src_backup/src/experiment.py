from argparse import ArgumentParser
from corpus import *
from lda import *

import cProfile
import pstats
from pstats import SortKey


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

    lda1 = LDA(corpus, args.T, args.S, args.optimize, args.output_dir)
    cProfile.runctx('lda1.inference()', globals(), locals(), filename="_py")
    pstats.Stats('_py').strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

    lda2 = LDA(corpus, args.T, args.S, args.optimize, args.output_dir)
    cProfile.runctx('lda2.cy_inference()', globals(), locals(), filename="_cy")
    pstats.Stats('_cy').strip_dirs().sort_stats(SortKey.TIME).print_stats(10)


if __name__ == '__main__':
    main()
