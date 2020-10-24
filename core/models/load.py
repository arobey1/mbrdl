import yaml
import torch
import torch.nn as nn
from apex import amp

from models.munit.networks import AdaINGen

def load_model(args, reverse=False):
    """Load MUNIT model and initialize with half-precision
    if args.half_precision flag is set.
    
    Params:
        args: Command line arguments for main.py.

    Returns:
        Model of natural variation as nn.Module instance.
    """

    def init_G(fname, reverse, args):
        """Load an MUNIT model of natural variation."""

        if args.setup_verbose is True and args.local_rank == 0:
            print(f'Loading MUNIT model: {fname}')

        G = MUNITModelOfNatVar(fname, reverse=reverse, args=args).cuda()

        # save model to ONNX
        # if args.local_rank == 0:
        #     G = MUNITModelOfNatVar(fname, reverse=reverse, args=args)

        #     dummy_x = torch.rand(256, 3, 32, 32)
        #     dummy_delta = torch.randn(256, 2, 1, 1)
        #     torch.onnx.export(G, (dummy_x, dummy_delta), 'munit.onnx', verbose=True)

        if args.half_prec is True:
            G = amp.initialize(G, opt_level=args.apex_opt_level, verbosity=0).half()
        return G

    if len(args.model_paths) == 1:
        return init_G(args.model_paths[0], reverse, args)
    elif len(args.model_paths) > 1:
        Gs = [init_G(fname, reverse, args) for fname in args.model_paths]
        return CompositionModel(*Gs)
    else:
        raise ValueError('You must supply a path to a model of natural variation.')

class CompositionModel(nn.Module):

    def __init__(self, *args):
        """Class that composes multiple models of natural variation.
        
        Params:
            Gs: Instantiations of >1 models of natural variation. 
        """

        super(CompositionModel, self).__init__()
        self._Gs = args

    def forward(self, x, delta):
        """Forward pass through composed models of natural variation."""

        for G in self._Gs:
            x = G(x, delta)

        return x

class MUNITModelOfNatVar(nn.Module):
    def __init__(self, fname: str, reverse: bool, args: dict):
        """Instantiantion of pre-trained MUNIT model.
        
        Params:
            fname: File name of trained MUNIT checkpoint file.
            reverse: If True, returns model mapping from domain A-->B.
                otherwise, model maps from B-->A.
            args: train.py command line arguments.
        """

        super(MUNITModelOfNatVar, self).__init__()

        self._config = self.__get_config(args.config)
        self._fname = fname
        self._reverse = reverse
        self._gen_A, self._gen_B = self.__load()
        self.delta_dim = self._config['gen']['style_dim']


    def forward(self, x, delta):
        """Forward pass through MUNIT model of natural variation."""

        orig_content, _ = self._gen_A.encode(x)
        orig_content = orig_content.clone().detach().requires_grad_(False)
        new_x = self._gen_B.decode(orig_content, delta)

        return new_x

    def __load(self):
        """Load MUNIT model from file."""

        def load_munit(fname, letter):
            gen = AdaINGen(self._config[f'input_dim_{letter}'], self._config['gen'])
            gen.load_state_dict(torch.load(fname)[letter])
            return gen.eval()

        gen_A = load_munit(self._fname, 'a')
        gen_B = load_munit(self._fname, 'b')

        if self._reverse is False:
            return gen_A, gen_B     # original order
        return gen_B, gen_A         # reversed order

    @staticmethod
    def __get_config(path):
        """Load .yaml file as dictionary.
        
        Params:
            path: Path to .yaml configuration file.
        """

        with open(path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)