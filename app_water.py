import speech_recognition as sr
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from statistics import mode

Random1 = pickle.load(open('Random1.pickle', 'rb'))
Random2 = pickle.load(open('Random2.pickle', 'rb'))
Random3 = pickle.load(open('Random3.pickle', 'rb'))
Random4 = pickle.load(open('Random4.pickle', 'rb'))
Random5 = pickle.load(open('Random5.pickle', 'rb'))
tv = pickle.load(open('tv.pickle', 'rb'))



recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Adjusting noise ")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Recording for 4 seconds")
    recorded_audio = recognizer.listen(source, timeout=4)
    print("Done recording")

try:
    print("Recognizing the text")
    text = recognizer.recognize_google(
            recorded_audio, 
            language="en-US"
        )

    print("Decoded Text : {}".format(text))
    new_data = [text]
    new_vector =tv.transform(new_data)
    pred1 = Random1.predict(new_vector)
    pred2 = Random2.predict(new_vector)
    pred3 = Random3.predict(new_vector)
    pred4 = Random4.predict(new_vector)
    pred5 = Random5.predict(new_vector)
    cumulative_pred=[]
    for i in range(pred1.shape[0]):
        temp_pred_list=[]
        temp_pred_list.append(pred1[i])
        temp_pred_list.append(pred2[i])
        temp_pred_list.append(pred3[i])
        temp_pred_list.append(pred4[i])
        temp_pred_list.append(pred5[i])
        cumulative_pred.append(mode(temp_pred_list))
    print(cumulative_pred[0])

except Exception as ex:

    print(ex)


sr.Microphone.list_microphone_names()