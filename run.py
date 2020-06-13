### This is the basic image classifier code using transfer learning ###
from tensorflow.keras.applications.vgg19 import VGG19, decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def test(img, model):
    img = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

if __name__ == '__main__':
    #model = VGG19(weights='imagenet')                               # Use this if you don't have model not in your local
    model = load_model('pretrained weights/vgg19.h5')                # use this if you have model in your local
    test_image = 'images/2464903-1366x768.jpg'                       # pass the image you want to predict
    prediction = test(test_image, model)
    label = decode_predictions(prediction)
    label = label[0][0]
    print("Predicted object is " +str(label[1])+ " and it's score is " +str(label[2]*100))
