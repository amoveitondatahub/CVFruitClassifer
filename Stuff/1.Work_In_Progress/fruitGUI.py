from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model("fruitmodel.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_folder_path = './fruits-360/Test'
test_img_gen = test_datagen.flow_from_directory(test_data_folder_path, (100,100),batch_size=32, class_mode='categorical', shuffle=False)
dictOFfruits = test_img_gen.class_indices

size = 200, 200

root = Tk()
root.title('Fruit Classifier')
root.geometry("900x600")
root.resizable(0, 0)

def clear():
    imageLabel.destroy()
    showPath.destroy()
    prediction.destroy()

    clear['state'] = DISABLED
    browse['state'] = NORMAL

def open():
    global photo, imageLabel, showPath, prediction
    root.filename = filedialog.askopenfilename(title="Select a File", filetypes=(("JPG","*.jpg"),("JPEG","*.jpeg"),("PNG","*.png")))
    im = Image.open(root.filename)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save('display.png')
    photo = PhotoImage(file='display.png')
    showPath = Label(root, text=root.filename)
    imageLabel = Label(root, image=photo)
    imageLabel.image = photo

    showPath.configure(background="white")
    imageLabel.configure(background="white")

    showPath.pack()
    imageLabel.pack()

    browse['state'] = DISABLED
    clear['state'] = NORMAL

    predictImg = image.load_img(root.filename,target_size=(100,100))
    predictImg = image.img_to_array(predictImg)
    imgTransfrom = np.expand_dims(predictImg,axis=0)
    imgTransfrom = imgTransfrom/255
    res = model.predict_classes(imgTransfrom)
    resName = list(dictOFfruits.keys())[list(dictOFfruits.values()).index(res)] 

    prediction = Label(root,text="prediction: {},  {}".format(res, resName), background="white")
    prediction.pack(ipady=10)

heading = Label(root, text="Add an image", background="white")
browse = Button(root, text="Open File", command=open)
clear = Button(root, text="clear", command=clear)
clear['state'] = DISABLED

groupNames = Label(root, text="Group Members: Abdullah Anwar, Bilal Rizwan, Bilal Zubairi", background="white")

root.configure(background="white")
heading.config(font=("Arial", 40, "bold"))
browse.configure(background="white")
clear.configure(background="white")

heading.pack(ipady=50)
browse.pack()
clear.pack()
groupNames.place(rely=0.95, relx=0.5)

root.mainloop()
