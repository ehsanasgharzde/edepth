import torch
import logging
import traceback
import os

def export_onnx(model, dummy_input, export_path, input_names=None, output_names=None, opset_version=17):
    """Export a PyTorch model to ONNX format."""
    try:
        model.eval()
        input_names = input_names or ["input"]
        output_names = output_names or ["output"]
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'}}
        )
        logging.info(f"Model exported to ONNX at {export_path}")
    except Exception as e:
        logging.error(f"ONNX export failed: {e}\n{traceback.format_exc()}")
        raise

def export_torchscript(model, dummy_input, export_path):
    """Export a PyTorch model to TorchScript via tracing."""
    try:
        model.eval()
        traced = torch.jit.trace(model, dummy_input)
        traced.save(export_path)
        logging.info(f"Model exported to TorchScript at {export_path}")
    except Exception as e:
        logging.error(f"TorchScript export failed: {e}\n{traceback.format_exc()}")
        raise

def print_deployment_plan():
    print("For TensorRT: Use the exported ONNX file with NVIDIA TensorRT tools (trtexec, torch2trt, or ONNX GraphSurgeon).\n"
          "For DeepSpeed: Integrate DeepSpeed engine in your training/inference pipeline.\n"
          "See official docs for details and compatibility.") 