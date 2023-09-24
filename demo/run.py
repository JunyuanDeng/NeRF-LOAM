import os  # noqa
import sys  # noqa
sys.path.insert(0, os.path.abspath('src')) # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from parser import get_parser
import numpy as np
import torch
from nerfloam import nerfloam
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = get_parser().parse_args()
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)
    else:
        setup_seed(777)

    slam = nerfloam(args)
    slam.start()
    slam.wait_child_processes()