import tkinter as tk
import tkinter.filedialog as filedialog
import numpy as np
import keras
import os
import tensorflow as tf
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image

PATH = os.getcwd()

# config = tf.ConfigProto()
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 4} ) 
# config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type = 'BFC'
session = tf.Session(config=config)
keras.backend.set_session(session)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.IMAGE_SIZE = 512
        self.isOpen = False
        self.pack()
        self.load_saved_model()
        self.create_buttons()
        master.configure(background='gray')
        root.geometry("500x500")
        # root.
    def create_buttons(self):
        #open File button
        self.openFileButton = tk.Button(self)
        self.openFileButton["text"] = "Open File"
        self.openFileButton["command"] = self.open_file
        self.openFileButton.pack(side="left")

        #analyze File button
        self.analyzeButton = tk.Button(self)
        self.analyzeButton["text"] = "Analyze Image"
        self.analyzeButton["command"] = self.analyze_image
        self.analyzeButton.pack(side="left")

        # self.quit = tk.Button(self, text="QUIT", fg="red",command=root.destroy)
        # self.quit.pack(side="right")

    def open_file(self):
        root.filename =  filedialog.askopenfilename(initialdir = "~/Desktop",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        # print(root.filename)
        self.imageLabel = tk.Label(root)
        self.imageLabel.image = ImageTk.PhotoImage(Image.open(root.filename))
        # openImage = tk.Label(root,image=image)
        self.imageLabel['image'] = self.imageLabel.image
        self.imageLabel.pack()
        self.isOpen = True

        self.closeImageButton = tk.Button(self, fg="red")
        self.closeImageButton["text"] = "Close Image"
        self.closeImageButton["command"] = self.close_image
        self.closeImageButton.pack(side="right")        

    def close_image(self):
        self.closeImageButton.destroy()
        self.isOpen = False
        self.imageLabel.destroy()


    def analyze_image(self):
        if self.isOpen:
            print("Analyzing image")
            print(root.filename)
            img = image.load_img(root.filename, target_size=(self.IMAGE_SIZE,self.IMAGE_SIZE))
            img = np.array(img)
            img = img.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE,3)
            prediction = self.model.predict(img)
            # self.model.predictImage(root.filename)
            print("predicting: ",prediction)
        else:
            print("No image open")

    def load_saved_model(self):
        self.model = load_model('./models/CNN_weights3.h5')

root = tk.Tk()
app = Application(master=root)
app.mainloop()