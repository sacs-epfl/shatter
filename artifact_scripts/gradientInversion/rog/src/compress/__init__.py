from .compressor import *

compress_registry = {
    "uniform": UniformQuantizer,
    "topk": Topk,
    "chunking_layerwise_first": ChunkingLayerwiseFirst,
    "chunking_layerwise_random": ChunkingLayerwiseRandom,
    "qsgd": QsgdQuantizer,
}
