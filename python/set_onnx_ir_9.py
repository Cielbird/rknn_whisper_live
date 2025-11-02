import onnx

model = onnx.load("../model/whisper_encoder_base.onnx")
print("Original IR version:", model.ir_version)

# Set a specific IR version (e.g., 9)
model.ir_version = 9

onnx.save(model, "../model/whisper_encoder_base.onnx")
