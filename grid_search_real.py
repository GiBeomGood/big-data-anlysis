import argparse
import pickle
from itertools import product
from os import listdir

from communityDetection import load_network
from models import *
from utils.util import eval


def main():
    scores = dict()
    grid = {
        'data_path': [f'./real/{path}' for path in listdir('./real')],
        'epsilon': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        'strategy': ['prob', 'freq'],
        'use_coreness': [True, False]
    }

    for data_path, epsilon, strategy, use_coreness in product(*grid.values()):
        args = argparse.Namespace(
            networkFile=f'{data_path}/network.dat',
            communityFile=f'{data_path}/community.dat',
            communitySavePath=f'{data_path}/prediction.cmty',
        )
        graph = load_network(args.networkFile)
        labels = my_nlta(graph, epsilon, strategy, use_coreness, args.communityFile, max_epochs=20)
        score = eval(labels, args.communityFile)

        scores[f'{data_path.split('/')[-1]}-{epsilon}-{strategy}-{use_coreness}'] = score

    with open('./checkpoints/ours_with_coreness_real.pickle', 'wb') as f:
        pickle.dump(scores, f)


if __name__ == '__main__':
    main()