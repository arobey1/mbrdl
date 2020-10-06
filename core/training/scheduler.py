import collections
import copy

class Scheduler:
    def __init__(self, optimizer, args, tb, log):
        phases = [copy.deepcopy(p) for p in eval(args.phases) if 'lr' in p]
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])
        self.tb = tb
        self.log = log
        self.args = args

    def format_phase(self, phase):
        phase['ep'] = self.listify(phase['ep'])
        phase['lr'] = self.listify(phase['lr'])
        if len(phase['lr']) == 2: 
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['ep']
        if 'epoch_step' in phase: batch_curr = 0 # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr 
        step_size = (lr_end - lr_start)/step_tot
        return lr_start + step_curr * step_size
    
    def get_current_phase(self, epoch):
        for phase in reversed(self.phases): 
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')
            
    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1: return phase['lr'][0] # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot) 
        if self.current_lr == lr: return
        if ((batch_num == 1) or (batch_num == batch_tot)): 
            self.log.event(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.tb.log("sizes/lr", lr)
        self.tb.log("sizes/momentum", self.args.momentum)

    @staticmethod
    def listify(p=None, q=None):
        if p is None: p=[]
        elif not isinstance(p, collections.Iterable): p=[p]
        n = q if type(q)==int else 1 if q is None else len(q)
        if len(p)==1: p = p * n
        return p