"""
Module for utilities when using ONNX/RKNN models
"""

import onnxruntime

def init_model(model_path, target=None, device_id=None):
    """Init a ONNX or RKNN model with a path to the model file"""
    if model_path.endswith(".rknn"):
        # pylint: disable=import-error,import-outside-toplevel
        from rknn.api import RKNN

        # Create RKNN object
        model = RKNN()

        # Load RKNN model
        print("--> Loading model")
        ret = model.load_rknn(model_path)
        if ret != 0:
            print(f'Load RKNN model "{model_path}" failed!')
            exit(ret)
        print("done")

        # init runtime environment
        print("--> Init runtime environment")
        ret = model.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("done")

    elif model_path.endswith(".onnx"):
        model = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    return model


def release_model(model):
    """Clean up a ONNX or RKNN model"""
    if "rknn" in str(type(model)):
        model.release()
    elif "onnx" in str(type(model)):
        del model
    model = None
