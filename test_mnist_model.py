import torch
import pytest
from mnist_low_param import CNN
from torchsummary import summary
import io
import sys

# Run summary(model) once and capture its output
with torch.no_grad():
    model = CNN()
    captured_output = io.StringIO()
    sys.stdout = captured_output
    summary(model, (1, 28, 28))
    sys.stdout = sys.__stdout__
    MODEL_SUMMARY = captured_output.getvalue()

@pytest.fixture(scope="session")
def model_summary():
    """Fixture that returns the pre-computed model summary"""
    return MODEL_SUMMARY

def test_parameter_count(model_summary):
    # Extract total parameters from summary
    total_params = int(model_summary.split("Total params: ")[1].split("\n")[0].replace(",", ""))
    assert total_params < 20000, f"Model has {total_params} parameters, exceeding 20,000 limit"

def test_batch_normalization_usage(model_summary):
    # Check if BatchNorm layers are present in summary
    assert "BatchNorm2d" in model_summary, "BatchNorm layers not found in model summary"
    
    # Count occurrences of BatchNorm2d in summary
    bn_count = model_summary.count("BatchNorm2d")
    assert bn_count >= 2, f"Model should have at least 2 BatchNorm layers, found {bn_count}"

def test_dropout_usage(model_summary):
    # Check if Dropout layers are present in summary
    assert "Dropout" in model_summary, "Dropout layers not found in model summary"
    
    # Count occurrences of Dropout in summary
    dropout_count = model_summary.count("Dropout")
    assert dropout_count >= 2, f"Model should have at least 2 Dropout layers, found {dropout_count}"

def test_final_layer_architecture(model_summary):
    # Check if either GAP or FC is present in summary
    has_gap = "AdaptiveAvgPool2d" in model_summary
    has_fc = "Linear" in model_summary
    
    assert has_gap or has_fc, "Model must have either GAP (AdaptiveAvgPool2d) or FC (Linear) layer"

if __name__ == "__main__":
    pytest.main(["-v", __file__])