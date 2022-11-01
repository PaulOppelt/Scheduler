import torch.optim as optim
import torch.nn as nn

class linear_scheduler_warmup(nn.Module):
    r"""implement scheduler that handles the progression of the learning rate. In the warm-up
    phase the lr increases from a start value linearly to a maximum and afterwards linearly
    decreases to to a set end value
    Args:
        optimizer: fetch learning rate from and adjust
        start: after warm-up, the learning rate starts to decrease from the start value
        end: final learning rate
        warmup_start: first learning rate
        total_steps: number of training iterations and subsequent scheduler steps. len(trainloader)
        ratio: lenght total-steps / lenght warmup-steps
    """
    def __init__(self,
                 optimizer: nn.Module,
                 start: float = 1e-4,
                 end: float =  1e-6,
                 warmup_start: float = 1e-6,
                 total_steps: int = 24000,
                 ratio: float = 15):
        super(linear_scheduler_warmup, self).__init__()
        
        assert start > 0 and end > 0 and warmup_start > 0, "learning rate must be strictly positive."
        assert ratio > 1, "Ratio must be larger than one."
        
        if optimizer.param_groups[0]["lr"] != warmup_start:
            raise Warning(f"initial learning rate is changed to {warmup_start}")
            
        
        self.optimizer = optimizer
        self.end = end
        self.start = start
        self.warmup_start = warmup_start
        
        # set steps
        self.total_steps = total_steps
        self.warmup_steps = self.total_steps // ratio
        
        # calculate the slopes of the linear increase/ decrease
        self.sloap_train = (self.end - self.start) / (self.total_steps - self.warmup_steps)
        self.sloap_warmup = (self.start - self.warmup_start) / self.warmup_steps
        
        # set the first learning rate
        self.optimizer.param_groups[0]["lr"] = self.warmup_start
        self.calls = 0
    
        
    def step(self)->None:
        r"""adjusts the learning rate of the optimizer.
        """
        if self.calls < self.warmup_steps: 
            self.optimizer.param_groups[0]["lr"] += self.sloap_warmup
            self.calls +=1
        else:
            self.optimizer.param_groups[0]["lr"] += self.sloap_train
            self.calls +=1
