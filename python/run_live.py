"""
Main module for running LiveWhisper with RKNN or ONNX in command-line
"""

import argparse

from live import LiveWhisper
from whisper import EN_LANG_TOKEN, ZH_LANG_TOKEN
from util import read_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Python Demo", add_help=True)
    # basic params
    parser.add_argument(
        "--encoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--vad_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--audio_file", type=str, required=False, help="File to use as audio input"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="recognition task, could be en or zh"
    )
    parser.add_argument(
        "--target", type=str, default="rk3588", help="target RKNPU platform"
    )
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    args = parser.parse_args()

    # Set inputs
    if args.task == "en":
        vocab_path = "../model/vocab_en.txt"
        task_code = EN_LANG_TOKEN
    elif args.task == "zh":
        vocab_path = "../model/vocab_zh.txt"
        task_code = ZH_LANG_TOKEN
    else:
        print(
            "\n\033[1;33mCurrently only English or Chinese recognition tasks are supported.",
            "Please specify --task as en or zh\033[0m",
        )
        exit(1)
    vocab = read_vocab(vocab_path)

    live = LiveWhisper(
        args.encoder_model_path,
        args.decoder_model_path,
        args.vad_model_path,
        args.target,
        args.device_id,
        vocab,
    )
    live.run(task_code=task_code, audio_file=args.audio_file)
