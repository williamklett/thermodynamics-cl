"""Generate arithmetic training data. Based on: https://github.com/karpathy/minGPT/tree/master/projects/adder"""

import omegaconf
import torch
from torch.utils.data import Dataset


def generate_arithmetic_problems(
    config: omegaconf.DictConfig,
) -> dict[str, torch.Tensor]:
    """
    Generate all possible addition problems with ndigit digits, split into train and test sets.
    Args:
        ndigit: Number of digits in the operands
    Returns:
        Dict with keys "train" and "test" containing the indices of the train and test sets.
    """
    ndigit = config.ndigit
    num = (10**ndigit)**2

    rng = torch.Generator()
    rng.manual_seed(1337)
    perm = torch.randperm(num, generator=rng)

    num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500

    return {"train": perm[num_test:], "test": perm[:num_test]}


class ArithmeticDataset(Dataset):
    """
    """

    def __init__(self, config, split):
        self.config = config

        ixes = generate_arithmetic_problems(self.config)
        self.ixes = ixes[split]

    def get_vocab_size(self) -> int:
        # digits 0..9, plus '-' token (10) for subtraction
        return 10

    def get_block_size(self) -> int:
        # a,b,a+b
        # +1 (due to potential carry overflow)
        # -1 (last digit doesn't ever plug back)
        return 3 * self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y
