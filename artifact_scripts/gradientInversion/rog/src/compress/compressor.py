import numpy as np
import torch


class UniformQuantizer:
    def __init__(self, config):
        self.quantbound = config.quant_level - 1

    def compress(self, arr):
        """
        quantize a given arr array with unifrom quant.
        """
        max_val = torch.max(arr.abs())
        sign_arr = arr.sign()
        quantized_arr = (arr / max_val) * self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).to(torch.int)

        quantized_set = dict(
            max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr
        )
        return quantized_set

    def decompress(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = (
            quantized_set["max_val"] / self.quantbound * quantized_set["signs"]
        )
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr


class SignSGDCompressor:
    def __init__(self, config):
        pass

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encoded_tensor = (tensor >= 0).to(torch.float)
        return encoded_tensor

    def decompress(self, tensor):
        """Decode the signs to float format"""
        decoded_tensor = tensor * 2 - 1
        return decoded_tensor


class Topk:

    def __init__(self, config):
        self.sparsity = config.sparsity

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        k = np.ceil(tensor.numel() * (1 - self.sparsity)).astype(int)
        top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
        tensor_masked = (tensor.abs() > -top_k_element) * tensor

        return tensor_masked

    def decompress(self, tensor):
        """Return the original tensor"""
        return tensor


class ChunkingLayerwiseFirst:

    def __init__(self, config):
        self.num_chunks = config.num_chunks

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with chunking and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        chunk_id = kwargs["chunk_id"] if "chunk_id" in kwargs else 0
        print("chunk_id", chunk_id)
        chunk_size = tensor.numel() // self.num_chunks
        start_idx = chunk_id * chunk_size
        end_idx = start_idx + chunk_size
        if chunk_id == self.num_chunks - 1:
            end_idx = tensor.numel()

        original_shape = tensor.shape
        tensor = tensor.flatten()
        tensor_masked = torch.zeros_like(tensor)
        tensor_masked[start_idx:end_idx] = tensor[start_idx:end_idx]
        tensor_masked = tensor_masked.reshape(original_shape)

        return tensor_masked

    def decompress(self, tensor):
        """Return the original tensor"""
        return tensor


class ChunkingLayerwiseRandom:

    def __init__(self, config):
        self.num_chunks = config.num_chunks

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        chunk_id = kwargs["chunk_id"] if "chunk_id" in kwargs else 0
        layer_id = kwargs["layer_id"] if "layer_id" in kwargs else 0
        chunk_id = (chunk_id + layer_id) % self.num_chunks
        print("chunk_id", chunk_id)

        total_elements = tensor.numel()
        generator = torch.Generator()
        generator.manual_seed(91)
        indices = torch.randperm(total_elements, generator=generator)

        chunk_size = total_elements // self.num_chunks
        start_idx = chunk_id * chunk_size
        end_idx = start_idx + chunk_size
        if chunk_id == self.num_chunks - 1:
            end_idx = total_elements

        indices = indices[start_idx:end_idx]
        print("number of indices selected", indices.shape, " out of ", total_elements)

        original_shape = tensor.shape
        tensor = tensor.flatten()
        tensor_masked = torch.zeros_like(tensor)
        tensor_masked[indices] = tensor[indices]
        tensor_masked = tensor_masked.reshape(original_shape)

        return tensor_masked

    def decompress(self, tensor):
        """Return the original tensor"""
        return tensor


class QsgdQuantizer:
    def __init__(self, config):
        self.quantlevel = config.quant_level
        self.quantbound = config.quant_level - 1

    def compress(self, arr):
        # norm = arr.norm()
        norm = torch.max(arr.abs())
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_()
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = lower_level + is_upper_level
        quantized_arr = torch.round(new_level)

        sign = arr.sign()
        quantized_set = dict(norm=norm, signs=sign, quantized_arr=quantized_arr)

        return quantized_set

    def decompress(self, quantized_set):
        coefficients = quantized_set["norm"] / self.quantbound * quantized_set["signs"]
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr
