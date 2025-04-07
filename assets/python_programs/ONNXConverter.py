import os
import tensorflow as tf
import tf2onnx
import onnx

model_dir = "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/models/"
model = tf.keras.models.load_model(os.path.join(model_dir, 'texture_discriminator_for_onnx.h5'))

spec = (tf.TensorSpec((None, 256, 256,1), tf.float32, name="input"),)

output_dir = "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/models"
output_path = os.path.join(output_dir, "texture_discriminator.onnx")

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)