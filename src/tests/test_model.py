import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TcgaNet


@pytest.fixture
def model():
    return TcgaNet(input_dim=30865, hidden_dim=128, output_dim=5, dropout=0.3)


def test_output_shape_batch(model):
    # 25 sample test with 30865 gene expression data
    dummy = torch.randn(25, 30865)
    output = model(dummy)
    # 25 samples and their 5 class prediction
    assert output.shape == torch.Size([25, 5])


def test_output_shape_single(model):
    # 1 sample test with 30865 gene expression data
    dummy = torch.randn(1, 30865)
    output = model(dummy)
    # 1 samples and their 5 class predictions
    assert output.shape == torch.Size([1, 5])

def test_different_hidden_dims():
    # test if hidden dim breaks on change
    model = TcgaNet(input_dim=30865, hidden_dim=256, output_dim=5, dropout=0.3)
    dummy = torch.randn(1, 30865)
    output = model(dummy)
    assert output.shape == torch.Size([1, 5])

def test_dropout_train_mode():
    # test if dropout impacting the results correctly
    model = TcgaNet(input_dim=30865, hidden_dim=128, output_dim=5, dropout=0.3)
    model.train()
    dummy = torch.randn(1, 30865)
    output1 = model(dummy)
    output2 = model(dummy)
    # two passes through same input should give different results
    assert not torch.equal(output1, output2)

def test_dropout_eval_mode():
    # test if dropout during eval is correctly working
    model = TcgaNet(input_dim=30865, hidden_dim=128, output_dim=5, dropout=0.3)
    # disable dropoff
    model.eval()
    dummy = torch.randn(1, 30865)
    with torch.no_grad():
        output1 = model(dummy)
        output2 = model(dummy)
    assert torch.equal(output1, output2)