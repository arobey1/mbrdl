import pandas as pd
import os

class Saver:
    def __init__(self, args, n_epochs):
        self._top1 = []
        self._top5 = []
        self._args = args
        self._n_epochs = n_epochs

    def update(self, top1, top5):
        self._top1.append(top1)
        self._top5.append(top5)
        self.save()

    def save(self):
        alg = self.get_alg()
        epochs = range(self._args.start_epoch, self._n_epochs)

        columns = ['Epoch', 'Top1-Accuracy', 'Top5-Accuracy', 'Algorithm']
        data = list(zip(
            epochs, self._top1, self._top5, [alg for _ in range(len(epochs))]
        ))
        df = pd.DataFrame(data, columns=columns)

        fname = os.path.join(self._args.save_path, f'{alg}-results.pkl')
        df.to_pickle(fname)

    def get_alg(self):

        if self._args.mrt is True:
            return f'MRT-{self._args.k}'
        elif self._args.mda is True:
            return f'MDA-{self._args.k}'
        elif self._args.mat is True:
            return f'MAT-{self._args.k}'
        elif self._args.pgd is True:
            return 'PGD'
        else:
            return 'Baseline'