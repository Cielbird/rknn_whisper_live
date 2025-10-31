import sys
from rknn.api import RKNN

DEFAULT_QUANT = False


def parse_arg():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] [dynamic shape preset(optional)]".format(
                sys.argv[0]
            )
        )
        print(
            "       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]"
        )
        print(
            "       dtype choose from [fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]"
        )
        print(
            "       dynamic shape preset choose from [whisper_decoder]"
        )
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ["i8", "u8", "fp"]:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ["i8", "u8"]:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = model_path.replace(".onnx", ".rknn")

    dynamic_inputs = None
    if len(sys.argv) > 5:
        dynamic_shape_preset = sys.argv[5]
        if dynamic_shape_preset == "whisper_decoder":
            max_tokens = 12 # TODO i would have liked up to 448 max tokens but oh well
            # TODO the second shape shouldn't be magic numbers: i got it from the ONNX file itself
            dynamic_inputs = [[[1, n + 1], [1, 1000, 512]] for n in range(0, max_tokens, 1)]

    return model_path, platform, do_quant, output_path, dynamic_inputs


if __name__ == "__main__":
    model_path, platform, do_quant, output_path, dynamic_inputs = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print("--> Config model")
    rknn.config(target_platform=platform, dynamic_input=dynamic_inputs)
    print("done")

    # Load model
    print("--> Loading model")
    if dynamic_inputs is None:
        ret = rknn.load_onnx(model=model_path)
    else:
        ret = rknn.load_onnx(model=model_path, input_size_list=dynamic_inputs[-1])
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    # Release
    rknn.release()
