import torch


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(x)
    return 1 - t * t


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def relu_prime(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()
