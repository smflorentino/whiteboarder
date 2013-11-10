import cv2
import numpy as np

from PIL import Image
from Tkinter import Tk, Text, PhotoImage
from ttk import Frame, Button, Style, Label
import LinkedListRep as _LL_


class gui(Frame):
    def __init__(self,master=Tk()):
        Frame.__init__(self,master)
        self.grid()
        self.image = Image.open("test.jpg")
        self.photo = PhotoImage(self.image)
        self.label = Label(master,image=self.photo)
        self.label.image = photo
        self.label.pack()



        






g = gui()
g.pack()
g.mainloop()
