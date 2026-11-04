"""Compare the CUDA block with autograd, including symmetric image boundaries."""

import pytest
import torch
import torch.nn.functional as F

from msd_pytorch.msd_block import MSDBlock
from msd_pytorch.conv2d import conv2d, conv2d_relu
from msd_pytorch.conv3d import conv3d, conv3d_relu


@pytest.fixture(autouse=True)
def ieee_convolutions():
    # Compare with the kernels' full float32 arithmetic rather than cuDNN TF32.
    previous = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cudnn.conv.fp32_precision = previous


def symmetric_pad(input, dilation):
    # The MSD kernels repeat the edge pixel; F.pad(mode="reflect") does not.
    for dim in range(2, input.ndim):
        size = input.shape[dim]
        indices = torch.arange(-dilation, size + dilation, device=input.device)
        indices = torch.where(indices < 0, -indices - 1, indices)
        indices = torch.where(indices >= size, 2 * size - indices - 1, indices)
        input = input.index_select(dim, indices)
    return input


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("relu", [False, True])
def test_reusable_convolution_buffer(ndim, relu):
    torch.manual_seed(8)
    function = {(2, False): conv2d, (2, True): conv2d_relu,
                (3, False): conv3d, (3, True): conv3d_relu}[ndim, relu]
    input = torch.randn(1, 2, *([4] * ndim), device="cuda", dtype=torch.float64,
                        requires_grad=True)
    weight = torch.randn(2, 2, *([3] * ndim), device="cuda", dtype=torch.float64,
                         requires_grad=True)
    bias = torch.randn(2, device="cuda", dtype=torch.float64, requires_grad=True)
    buffer = torch.empty_like(input)
    conv = F.conv2d if ndim == 2 else F.conv3d
    for _ in range(2):
        actual = function(input, weight, bias, buffer, 1, 1)
        reference = conv(symmetric_pad(input, 1), weight, bias)
        if relu:
            reference = reference.relu()
        torch.testing.assert_close(actual, reference)
        assert actual.data_ptr() == buffer.data_ptr()
        gradients = torch.autograd.grad(actual.sum(), (input, weight, bias))
        reference_gradients = torch.autograd.grad(reference.sum(), (input, weight, bias))
        for gradient, expected in zip(gradients, reference_gradients):
            torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("frozen", [None, 0, 2])
def test_block_matches_autograd(ndim, dtype, frozen):
    torch.manual_seed(12)
    block = MSDBlock(2, [1, 2, 3], width=2, ndim=ndim).cuda().to(dtype)
    if frozen is not None:
        getattr(block, f"weight{frozen}").requires_grad_(False)

    input = torch.randn(2, 2, *([3, 4, 5][:ndim]), device="cuda", dtype=dtype,
                        requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    bias = block.bias.detach().clone().requires_grad_()
    weights = [getattr(block, f"weight{i}").detach().clone().requires_grad_(i != frozen)
               for i in range(3)]
    reference = reference_input
    conv = F.conv2d if ndim == 2 else F.conv3d
    for i, (dilation, weight) in enumerate(zip(block.dilations, weights)):
        output = conv(symmetric_pad(reference, dilation), weight,
                      bias[2 * i:2 * (i + 1)], dilation=dilation).relu()
        reference = torch.cat((reference, output), dim=1)

    actual = block(input)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    reference.backward(gradient)
    torch.testing.assert_close(input.grad, reference_input.grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(block.bias.grad, bias.grad, atol=2e-5, rtol=2e-4)
    for i, weight in enumerate(weights):
        actual_grad = getattr(block, f"weight{i}").grad
        if i == frozen:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, weight.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("ndim", [2, 3])
def test_non_default_cuda_stream(ndim):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        block = MSDBlock(1, [1, 2], ndim=ndim).cuda()
        input = torch.randn(1, 1, *([4] * ndim), device="cuda", requires_grad=True)
        output = block(input)
        output.square().mean().backward()
    stream.synchronize()
    assert torch.isfinite(output).all()
    assert torch.isfinite(input.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in block.parameters())
