import json # need to copy it to part 2
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
# import libraries
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', help='image path', action="store", type = str)
parser.add_argument('--model', default='./model.h5', help='checkpoint file path/name', action="store", type = str)
parser.add_argument('--top_k', default=5, help='return top K most likely classes', dest="top_k", action="store", type=int)
parser.add_argument('--category_names', default='label_map.json', help='mapping the categories to real names', dest="category_names", action="store")

arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

def process_image(image_num):
    image_size=224
    p_image= tf.convert_to_tensor(image_num, dtype=tf.float32)
    p_image= tf.image.resize(p_image,(image_size,image_size))
    p_image /=255
    
    return p_image

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_test = process_image(test_image)
    expanded_image = np.expand_dims(processed_test, axis=0)
    
    predicted = model.predict(expanded_image)
    predicted = predicted.tolist()
    probs, labels = tf.math.top_k(predicted, k=top_k)
    probs = probs.numpy().tolist()[0]
    labels = labels.numpy().tolist()[0]
    
    return probs, labels

if __name__ == '__main__':
    
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    loaded_model= tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})

    probs, classes = predict(image_path,loaded_model,topk)
    
    name= [class_names[str(int(label)+1)]for label in classes]

    print('probs',probs)
    print('classes',classes)
    print('label names',name)    