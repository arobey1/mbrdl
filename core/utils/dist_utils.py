import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import multiprocessing
import os

def whoami(args):
    """Determines if current rank is master and/or local rank 0.
    
    Params:
        args: Command line arguments for main.py.
    """

    is_master = (not args.distributed) or (env_rank() == 0)
    is_rank0 = args.local_rank == 0

    return is_master, is_rank0

def env_world_size(): 
    """World size for distributed training.
    Is set in torch.distributed.launch as args.nproc_per_node * args.nnodes
    see: https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py"""

    return int(os.environ['WORLD_SIZE'])

def env_rank(): 
    """Local rank of each GPU used in distributed training.
    Is set in torch.distributed.launch as args.nproc_per_node * args.node_rank + local_rank
    see: https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py"""

    return int(os.environ['RANK'])

def setup_dist_backend(args, set_threads=False, thread_choice=None):
    """Sets up backend/environment for distributed training.

    Params:
        args: Command line args for main.py.
        thread_choice: How to choose number of OMP threads used.
    """

    def setup_print(s, **kwargs):
        if args.setup_verbose is True:
            print(s, **kwargs)

    # assumes all data will have (roughly) the same dimensions
    cudnn.benchmark = True

    # choose environment variable OMP_NUM_THREADS
    # see: https://github.com/pytorch/pytorch/pull/22501
    if set_threads is True:
        if thread_choice is None:
            os.environ['OMP_NUM_THREADS'] = str(1)
        elif thread_choice == 'torch_threads':
            os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())
        elif thread_choice == 'multiproc':
            n_threads = (int)(multiprocessing.cpu_count() / os.environ['WORLD_SIZE'])
            os.environ['OMP_NUM_THREADS'] = str(n_threads)

    if args.distributed is True:
        if args.local_rank == 0:
            setup_print('Setting up distributed process group...')

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url, 
            world_size=env_world_size()
        )

        # make sure there's no mismatch between world sizes
        assert(env_world_size() == torch.distributed.get_world_size())
        setup_print(f"\tSuccess on process {args.local_rank}/{torch.distributed.get_world_size()}")
        
def reduce_tensor(tensor): 
    return sum_tensor(tensor) / env_world_size()

def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def sync_processes(args):
    """Perform a simple reduce operation to sync all processes.
    
    Params:
        args: Command line args for main.py.
    """

    tensor = torch.tensor([1.0]).float().cuda()
    rt = sum_tensor(tensor)

    if args.local_rank == 0 and args.setup_verbose:
        print(f'Gave tensor = {tensor.item()} to each process.  Summed results: {rt.item()}')
