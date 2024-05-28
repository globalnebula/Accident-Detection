import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="tf_lite_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape'][1:3]

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def visualize_feature_maps(interpreter, image_path):
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    tensor_details = interpreter.get_tensor_details()
    conv_layer_indices = [index for index, layer in enumerate(tensor_details) if 'conv' in layer['name']]
    
    for i, layer_index in enumerate(conv_layer_indices):
        activation = interpreter.get_tensor(layer_index)
        print(f"Layer {i+1}: {tensor_details[layer_index]['name']}")
        print(f"Activation shape: {activation.shape}")
        
        num_filters = activation.shape[-1]
        plt.figure(figsize=(16, 2))
        for j in range(num_filters):
            if len(activation.shape) == 4:
                plt.subplot(1, num_filters, j+1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')  # Select the first sample (batch size 1)
                plt.title(f'Filter {j+1}')
                plt.axis('off')
            else:
                plt.plot(activation[0])
                plt.title(f'Layer {i+1}: {tensor_details[layer_index]["name"]} - {num_filters} filters')
                plt.xlabel('Filter Index')
                plt.ylabel('Activation')
                plt.grid(True)
        plt.suptitle(f'Layer {i+1}: {tensor_details[layer_index]["name"]} - {num_filters} filters')
        plt.tight_layout()
        plt.show()





image_path_to_predict = 'test_33.jpg'
visualize_feature_maps(interpreter, image_path_to_predict)
