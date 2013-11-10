#This will be the UI with a video feed and a "take picture button."
#Upon taking a picture, the program will ask if the picture is ok
#with an option to retake or process image.

from Tkinter import Tk, Text, RIGHT, BOTH, RAISED, W, N, E, S
from ttk import Frame, Button, Style, Label


class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)#,background="white")   
         
        self.parent = parent
        self.parent.title("Centered window")
        self.pack(fill=BOTH, expand=1)
        self.centerWindow()
        self.initUI()

    def centerWindow(self):
      
        w = 290
        h = 150

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        
        x = (sw - w)/2
        y = (sh - h)/2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def initUI(self):
      
        self.parent.title("Whiteboarder")
        self.style = Style()
        self.style.theme_use("default")
        self.pack(fill=BOTH, expand=1)
        
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, pad=7)
        
        frame = Frame(self, relief=RAISED, borderwidth=1)
        #frame.pack(fill=BOTH, expand=1)

        lbl = Label(self)
        lbl.grid(sticky=W, pady=4, padx=15)

        area = Text(self)
        area.grid(row=1, column=0, columnspan=2, rowspan=4, 
            padx=15, sticky=E+W+S+N)

        lbtn = Button(self, text="Linked List")
        lbtn.grid(row=1, column=3, pady=4)

        abtn = Button(self, text="Array")
        abtn.grid(row=2, column=3, pady=4)

        gbtn = Button(self, text="Graph")
        gbtn.grid(row=3, column=3, pady=4)
        
        obtn = Button(self, text="Process")
        obtn.grid(row=5, column=3, padx=0) 
        
        
        #closeButton.pack(side=RIGHT, padx=5, pady=5)
        #okButton = Button(self, text="OK")
        #okButton.pack(side=RIGHT)
        

def main():
  
    root = Tk()
    root.geometry("300x200+300+300")
    app = Example(root)
    root.mainloop()  


if __name__ == '__main__':
    main()  
