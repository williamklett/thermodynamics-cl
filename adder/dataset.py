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
    Args:
        config: must have `ndigit` (problem difficulty)
        split: "train" or "test"
        max_ndigit: if provided, pad sequences to this size (for continual learning)
    """

    def __init__(self, config, split, max_ndigit: int = None):
        self.config = config
        self.ndigit = config.ndigit
        self.max_ndigit = max_ndigit if max_ndigit is not None else self.ndigit

        ixes = generate_arithmetic_problems(self.config)
        self.ixes = ixes[split]

    def get_vocab_size(self) -> int:
        # digits 0..9
        return 10

    def get_block_size(self) -> int:
        # Use max_ndigit for consistent sequence length
        # a,b,a+b
        # +1 (due to potential carry overflow)
        # -1 (last digit doesn't ever plug back)
        return 3 * self.max_ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.ndigit
        max_ndigit = self.max_ndigit
        
        # Recover the a + b problem from index
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx % nd
        c = a + b
        
        # Pad to max_ndigit format for consistent positional encoding
        astr = f'%0{max_ndigit}d' % a
        bstr = f'%0{max_ndigit}d' % b
        cstr = (f'%0{max_ndigit+1}d' % c)[::-1]  # reversed
        
        render = astr + bstr + cstr
        dix = [int(s) for s in render]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y[:max_ndigit*2-1] = -1  # mask input positions
        return x, y
