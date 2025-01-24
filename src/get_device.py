import torch


def get_device():
    """
    Determine the best available device for PyTorch operations.

    Returns:
        str: Device string, one of:
            - "cuda" if NVIDIA CUDA GPU is available
            - "xpu" if Intel IPU is available
            - "hip" if AMD ROCm is available
            - "mps" if Apple Metal (M1/M2) acceleration is available
            - "cpu" if no hardware acceleration is available
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif hasattr(torch, "hip") and torch.hip.is_available():
        return "hip"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
