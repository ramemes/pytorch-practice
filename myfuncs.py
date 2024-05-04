import torch

def reshape_with_fill(tensor, new_shape):
    #reshapes tensor like torch.tensor.reshape(new_shape)
    #if there are not enough values in tensor, remaining values will be filled with 0's
    #if there are too many values in tensor, the excess will be cut off
    result = torch.zeros(new_shape, dtype = tensor.dtype, device=tensor.device)
    num_elements = min(result.numel(),tensor.numel())

    result.view(-1)[:num_elements] = tensor.view(-1)[:num_elements]
    return result

    #make function able to cut out from start and end (keep middle)
    #skip out specific tensors or every other tensor to retain wholistic tensor