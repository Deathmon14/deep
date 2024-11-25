import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2

model = VGG16(weights='imagenet')
model.summary()

img_path = '/content/download.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = np.expand_dims(image.img_to_array(img), axis=0)
x = preprocess_input(x)

filters = model.get_layer('block1_conv1').get_weights()[0]
filters = (filters - filters.min()) / (filters.max() - filters.min())

plt.figure(figsize=(10, 10))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(filters[:, :, :, i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle('ConvNet Filters', fontsize=16)
plt.show()

activation_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
activations = activation_model.predict(x)
plt.figure(figsize=(10, 10))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(activations[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Activations from Layer 2', fontsize=16)
plt.show()

def compute_gradcam(model, img_array, layer_name, class_idx):
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        loss = prediction[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return heatmap / np.max(heatmap)

heatmap = compute_gradcam(model, x, 'block5_conv3', 386)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.suptitle('Grad-CAM Heatmap', fontsize=16)
plt.show()
