import tkinter as tk
import tkinter.filedialog as filedialog
import numpy as np
import os
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image

PATH = os.getcwd()
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.load_saved_model()
        self.create_buttons()
        master.configure(background='black')
        self.IMAGE_SIZE = 512

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

        self.quit = tk.Button(self, text="QUIT", fg="red",command=root.destroy)
        self.quit.pack(side="right")

    def open_file(self):
        root.filename =  filedialog.askopenfilename(initialdir = "~/Desktop",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        # print(root.filename)
        image_label = tk.Label(root)
        image_label.image = ImageTk.PhotoImage(Image.open(root.filename))
        # openImage = tk.Label(root,image=image)
        image_label['image'] = image_label.image
        image_label.pack()

    def analyze_image(self):
        print("Analyzing image")
        print(root.filename)
        img = image.load_img(root.filename, target_size=(self.IMAGE_SIZE,self.IMAGE_SIZE))
        img = np.array(img)
        img = img.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE,3)
        prediction = self.model.predict(img)
        # self.model.predictImage(root.filename)
        print("predicting: ",prediction)

    def load_saved_model(self):
        self.model = load_model('./models/CNN_weights.h5')

root = tk.Tk()
app = Application(master=root)
app.mainloop()