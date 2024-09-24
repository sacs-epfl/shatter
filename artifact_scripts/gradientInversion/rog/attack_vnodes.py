import torch
from main import main
import sys

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_fedavg.yaml"
    clients = sys.argv[2] if len(sys.argv) > 2 else 1
    chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(config_file, manual_seed=seed, client=clients, chunks=chunks)
