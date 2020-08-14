import argparse
import time

from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter('model.tflite',
  experimental_delegates=[load_delegate('libedgetpu.so.1', {"device": "pci:0"})])

interpreter.allocate_tensors()

#load image
im = Image.open("RGB-3.jpg")
width, height = im.size
width = int(width/2)
height = int(height/2)
im = im.resize(width,height)
img_array = np.array(im).reshape(-1, height, width, 3)
img_array = np.float32(img_array / 255.)

#set up model input/output
def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data

def output_tensor(interpreter):
  """Returns dequantized output tensor."""
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())
  # scale, zero_point = output_details['quantization']
  return output_data


set_input(interpreter, img_array)
start = time.perf_counter()
interpreter.invoke()
inference_time = time.perf_counter() - start
output_data = get_output(interpreter)
print('%.1fms' % (inference_time * 1000))
