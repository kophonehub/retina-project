import os.path
import tensorflow as tf
import streamlit as st
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    st.header('Diabetic Retinopathy Classifier')

    st.markdown('A simple web application for grading severity of diabetic retinopathy.')

    file_uploaded = st.file_uploader('Please upload your image dataset', type=("jpg", "png", "jpeg"))

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        print(file_uploaded.name)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        with open(os.path.join(".", file_uploaded.name), "wb") as f:
            f.write(file_uploaded.getbuffer())

        st.success("Saved")
    class_btn = st.button("Classify")

    if class_btn:
        if file_uploaded is None:
            st.write('Invalid command.Please upload an image')

        else:
            with st.spinner('Model Working.....'):
                plt.imshow(image)
                plt.axis("off")
                image_name = []
                classes = []
                image_name.append(file_uploaded.name)
                prob = import_and_predict(image)
                class_value = np.argmax(prob, axis=1)
                classes.append(class_value)
                st.success('Classified')
                if str(class_value[0]) == "1":
                    st.write('Diagnosed as "Non-Proliferative Diabetic Retinopathy".')
                else:
                    st.write('Diagnosed as "Proliferative Diabetic Retinopathy".')
                # st.write("Diabetic Retinopathy image grade is", str(class_value[0]))


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        checkshape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if checkshape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


IMG_SIZE = 224


def load_ben_color(path, sigmaX=10):
    # image = cv2.imread(path)
    image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def import_and_predict(image):
    # image = np.array(image)
    # image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image,(224,224))
    # image = np.array(image) / 255.0
    # new_model = tf.keras.models.load_model('./CNN_Model.h5')
    # predict = new_model.predict(np.array([image]))
    # return predict

    # model = tf.keras.models.load_model('./CNN_Model.h5')
    image = np.array(image)
    image = load_ben_color(image, sigmaX=80)
    new_model = tf.keras.models.load_model('./VGG16.h5')
    predict = new_model.predict(np.array([image]))
    return predict


main()
