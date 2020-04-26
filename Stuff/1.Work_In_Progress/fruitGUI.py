from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from random import seed
from random import randint
from os import listdir
from os.path import isfile, join
from PIL import ImageTk,Image
import keras
from keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

model = load_model("fruitmodel.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_folder_path = './fruits-360/Test'
test_img_gen = test_datagen.flow_from_directory(test_data_folder_path, (100,100),batch_size=32, class_mode='categorical', shuffle=False)
dictOFfruits = test_img_gen.class_indices

size = 300, 300

root = Tk()
root.title('Fruit Classifier')
root.geometry("900x600")

def popup(resPath, imgPath):
    response = messagebox.askyesno("prediction test", "is the prediction correct?")
    if(not response):
        imageLabel.destroy()
        showPath.destroy()
        prediction.destroy()
        compLabel.destroy()

        clear['state'] = DISABLED
        browse['state'] = NORMAL
    else:
        resImg = image.load_img(resPath)
        originalImg = image.load_img(imgPath,target_size=(700,700))

        originalImg = np.array(originalImg)
        resImg = np.array(resImg)

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(resImg,None)
        kp2, des2 = orb.detectAndCompute(originalImg,None)


        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        #Do Not Draw first 25 matches.
        reeses_matches = cv2.drawMatches(resImg,kp1,originalImg,kp2,matches,None,flags=0)

        graphWin = Toplevel()
        graphWin.geometry("900x600")

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(reeses_matches,cmap='gray')

        canvas = FigureCanvasTkAgg(fig, master=graphWin)
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def clear():
    imageLabel.destroy()
    showPath.destroy()
    prediction.destroy()
    compLabel.destroy()

    clear['state'] = DISABLED
    browse['state'] = NORMAL

def open():
    global photo, imageLabel, compLabel, showPath, prediction
    pictures = Frame(root)

    root.filename = filedialog.askopenfilename(title="Select an image", filetypes=(("JPG","*.jpg"),("JPEG","*.jpeg"),("PNG","*.png")))
    im = Image.open(root.filename)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save('display.png')
    photo = PhotoImage(file='display.png')
    
    showPath = Label(root, text=root.filename)
    imageLabel = Label(pictures, image=photo)
    imageLabel.image = photo

    predictImg = image.load_img(root.filename,target_size=(100,100))
    predictImg = image.img_to_array(predictImg)
    imgTransfrom = np.expand_dims(predictImg,axis=0)
    imgTransfrom = imgTransfrom/255
    res = model.predict_classes(imgTransfrom)
    resName = list(dictOFfruits.keys())[list(dictOFfruits.values()).index(res)] 

    seed(1)
    value = randint(0, 119)

    resPath = './fruits-360/Training/'+resName

    fileName = [f for f in listdir(resPath) if isfile(join(resPath, f))] 
    randomImg = np.random.randint(0,len(fileName))
    imgName = fileName[randomImg] 
    resPath = resPath + "/" + imgName

    cIm = Image.open(resPath)
    cIm.thumbnail(size, Image.ANTIALIAS)
    cIm.save('compare.png')
    comp = PhotoImage(file='compare.png')
    compLabel = Label(pictures, image=comp)
    compLabel.image = comp

    prediction = Label(root,text="prediction: {},  {}".format(res, resName), background="white")

    showPath.configure(background="white")
    pictures.configure(background="white")
    imageLabel.configure(background="white")
    compLabel.configure(background="white")
    prediction.config(font=("Arial", 24, "bold"))

    showPath.pack()
    pictures.pack()
    imageLabel.pack(ipadx= 5, side=LEFT)
    compLabel.pack(ipadx= 5, side=RIGHT)
    prediction.pack(ipady=50)

    browse['state'] = DISABLED
    clear['state'] = NORMAL

    popup(resPath, root.filename)



heading = Label(root, text="Fruit classification using CV", background="white")

buttons = Frame(root)
browse = Button(buttons, text="Open File", command=open)
clear = Button(buttons, text="clear", command=clear)
clear['state'] = DISABLED

groupNames = Label(root, text="Group Members: Abdullah Anwar, Bilal Rizwan, Bilal Zubairi", background="white")

root.configure(background="white")
buttons.configure(background="white")
heading.config(font=("Arial", 32, "bold"))
browse.configure(background="white")
clear.configure(background="white")

heading.pack(ipady=5)
buttons.pack(ipady=5)
browse.pack(side=LEFT)
clear.pack(side=RIGHT)
groupNames.place(rely=0.95, relx=0.5)

root.mainloop()
