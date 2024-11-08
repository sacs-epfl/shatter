import sys

import torch
from main import main

if __name__ == "__main__":
    manual_seed = 0
    torch.manual_seed(manual_seed)
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_fedavg.yaml"
    clients = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(config_file, manual_seed=manual_seed, clients=clients)
