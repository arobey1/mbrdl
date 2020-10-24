import pandas as pd
import os

def save_eval_df(top1, top5, args):
    """Save dataframe with results after evaulation."""

    df = pd.DataFrame(
            data=list(zip([top1], [top5], [args.architecture], [args.source_of_nat_var])),
            columns=['Top1-Accuracy', 'Top5-Accuracy', 'Architecture', 'Challenge']
        )
    fname = os.path.join(args.save_path, 'results.pkl')
    df.to_pickle(fname)

class Saver:
    def __init__(self, args, n_epochs):
        """Saves top1 and top5 accuracies during training.
        
        Params:
            args: Command line arguments.
            n_epochs: Number of epochs for training.
        """

        self._top1, self._top5 = [], []
        self._args = args
        self._n_epochs = n_epochs

    def update(self, top1, top5):
        """Update top1 and top5 lists and save updated DataFrame.
        
        Params:
            top1: Top1 accuracy from current epoch.
            top5: Top5 accuracy from current epoch.
        """

        self._top1.append(top1)
        self._top5.append(top5)
        self.save()

    def save(self):
        """Save current top1/top5 accuracies to a DataFrame."""

        alg = self.get_alg()
        epochs = range(self._args.start_epoch, self._n_epochs)

        # columns and data to be saved in DataFrame
        columns = ['Epoch', 'Top1-Accuracy', 'Top5-Accuracy', 'Algorithm']
        data = list(zip(
            epochs,
            self._top1, 
            self._top5, 
            [alg for _ in range(len(epochs))]
        ))
        df = pd.DataFrame(data, columns=columns)

        # save DataFrame to file
        fname = os.path.join(self._args.save_path, f'{alg}-results.pkl')
        df.to_pickle(fname)

    def get_alg(self):
        """Get algorithm for DataFrame save name."""

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