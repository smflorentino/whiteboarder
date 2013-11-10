from Tkinter import Tk, Text, RIGHT,BOTH,RAISED, W, N, E, S, Menu, Canvas
import edgedetection
from ttk import Frame, Button, Style, Label
from StructureCanvas import StructureCanvas
import LinkedListRep as LL


class Interface(Frame):

    def __init__(self,master=Tk()):
        Frame.__init__(self,master)
        self.grid()
        self.llbutton = Button(self,text="Linked List", command = self.createLL)
        self.llbutton.grid()
        self.canvas = Canvas(master,bg="white",height=750,width=1000)
        self.canvas.pack()


    def createLL():
        LL.linked_list()
        print "Linked List chosen"
        
    def drawNode(self,coord,rad,val):
        self.canvas.create_oval((coord[0],coord[1],coord[0],coord[1]),state="normal",width=rad)
        self.canvas.create_text((coord[0],coord[1]),text=val)

    def drawArrow(self,src,dest):

        if src[0] > dest[0]:
            x1 = src[0] - 20
            x2 = dest[0] + 20
        elif src[0] < dest[0]:
            x1 = src[0] + 20
            x2 = dest[0] - 20
        if src[1] > dest[1]:
            y1 = src[1] - 20
            y2 = dest[1] + 20
        elif src[1] < dest[1]:
            y1 = src[1] + 20
            y2 = dest[1] - 20
        self.canvas.create_line(x1,y1,x2,y2,arrowshape="8 10 7", arrow="last")
        dx = dest[0] - src[0]
        dy = dest[1] - dest[0]
        m = dy/dx
        inv = (dx/dy)*-1
#        self.canvas.create_line(x2,y2,-m*5,inv*5)
#        self.canvas.create_line(x2,y2,m*5,-inv*5)




def printList(list):
    head = list
    coords = (head.x,head.y)
    ui.drawNode(coords,head.r,'%d'%head.datum)
    if head.next is not None:
        nextNode = head.next
        nextCoords = (nextNode.x,nextNode.y)
        ui.drawArrow(coords,nextCoords)
        head = head.next
        printList(head)
    



printList(edgedetection.mainToGUI())














