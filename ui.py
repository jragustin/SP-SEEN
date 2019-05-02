import tkinter as tk
import tkinter.filedialog as filedialog
import os
from PIL import Image, ImageTk
PATH = os.getcwd()
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_buttons()
        master.configure(background='black')

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

root = tk.Tk()
app = Application(master=root)
app.mainloop()