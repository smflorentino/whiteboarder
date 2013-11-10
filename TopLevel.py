#This will be the UI with a video feed and a "take picture button."
#Upon taking a picture, the program will ask if the picture is ok
#with an option to retake or process image.

from Tkinter import Tk, Text, RIGHT, BOTH, RAISED, W, N, E, S
from ttk import Frame, Button, Style, Label
import LinkedListRep as _LL_
import RedirectText
from StructureCanvas import StructureCanvas 



class UI(Frame):
            
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
        self.rowconfigure(4, weight=0)
        self.rowconfigure(5, pad=7)
        
        frame = Frame(self, relief=RAISED, borderwidth=1)
        #frame.pack()

        lbl = Label(self)
        lbl.grid(sticky=W, pady=5, padx=15)
        
        root = Tk()
        area = StructureCanvas(root)#Text(self)
        root.mainloop()
        area.grid(row=1, column=0, columnspan=2, rowspan= 6, 
            padx=15, sticky=E+W+S+N)
        #area.pack(side="top", fill="both", expand=True)
        
        def createLL():
            _LL_.linked_list()
            print "Linked List chosen"

        lbtn = Button(self, text="Linked List", command = createLL())
        lbtn.grid(row=1, column=3, padx = 2, pady=4)

        abtn = Button(self, text="Array")
        abtn.grid(row=2, column=3, padx = 2, pady=4)

        tbtn = Button(self, text="Tree")
        tbtn.grid(row=3, column=3, padx = 2, pady=4)

        ogbtn = Button(self, text="Other Graph")
        ogbtn.grid(row=4, column=3, padx = 2, pady=4)
        
        obtn = Button(self, text="Process")
        obtn.grid(row=7, column=3, pady=4)

        #closeButton.pack(side=RIGHT, padx=5, pady=5)
        #okButton = Button(self, text="OK")
        #okButton.pack(side=RIGHT)       

def main():
  
    root = Tk()
    root.geometry("300x200+300+300")
    app = UI(root)
    root.mainloop()  


if __name__ == '__main__':
    main()

