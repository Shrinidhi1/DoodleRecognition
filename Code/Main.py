from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from AirDoodle import airDoodle
from Detection import detection
from ImageMatching import imageMatch
from Predict import main

classes = ['Alarm clock',
 'Apple',
 'Birthday cake',
 'Butterfly',
 'Candle',
 'Ceiling fan',
 'Donut',
 'Door',
 'Eyeglasses',
 'T-shirt']

def adjustWindow(window):
    w = 1000
    h = 750
    ws = window.winfo_screenwidth()
    hs = window.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    window.geometry('%dx%d+%d+%d' % (w, h, x, y))
    window.resizable(False, False)

def prediction():
    predict = main()
    image = Image.open("intermediate/doodle.png")
    image.thumbnail((350, 350))
    img = ImageTk.PhotoImage(image)
    # img = ImageTk.PhotoImage(Image.open("intermediate/Doodle.png"))
    Label(window, text=("Prediction: " + classes[int(np.argmax(predict[0]))] + "         "), font=("Arial", 15)).place(x=300, y=350)
    Label(window, text=("Percent confidence: " + str(round(np.max(predict[0]) * 100, 2))), font=("Arial", 15)).place(x=300, y=390)
    label1 = Label(window, image=img).place(x=300, y=450)
    label1.image = img

def scanImage():
    detection()
    imageMatch()
    prediction()

def createDoodle():
    airDoodle()
    prediction()

window = Tk()
# window.geometry("400x400")
adjustWindow(window)
window.title("DOODLE RECOGNITION")
Label(window, text="DOODLE RECOGNITION", font=("Arial", 25)).place(x=280, y=150)
# Label(window, text="Enter your choice:", font=("Arial", 15)).place(x=280, y=200)
Button(window, text="1. Scan the hand drawn doodle image.", command=scanImage, font=("Arial", 15), height= 1, width=30).place(x=300, y=250)
Button(window, text="2. Create doodle image in air.", command=createDoodle, font=("Arial", 15), height= 1, width=30).place(x=300, y=300)
window.mainloop()