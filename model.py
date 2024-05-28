import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tf_lite_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Load and preprocess a test image
# Replace "test_image.jpg" with the path to your test image
image = tf.keras.preprocessing.image.load_img("test_33.jpg", target_size=(250, 250))
image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0  # Normalize pixel values to [0, 1]

# Run inference on the test image
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Extract feature maps (activations) from the desired intermediate layer
# For example, let's extract the activations from the first convolutional layer
layer_index = 0  # Index of the output tensor
activations = interpreter.get_tensor(output_details[layer_index]['index'])

# Print the shape of the extracted feature maps
print("Shape of feature maps:", activations.shape)
