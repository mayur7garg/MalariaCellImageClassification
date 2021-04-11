from tensorflow.image import resize_with_pad
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_image_data(image_path, img_size):
    img = load_img(image_path)
    img = resize_with_pad(img_to_array(img, dtype = 'uint8'), *img_size).numpy().astype('uint8')
    return img
    