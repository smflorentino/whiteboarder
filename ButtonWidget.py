import sys
from PyQt4 import QtGui
import LinkedListRep as LL

class ButtonWidget(QtGui.QWidget):

    def __init__(self):
        super(ButtonWidget, self).__init__()

        self.initUI()

    def initUI(self):

        def createLL():
            LL.linked_list()
            print "Linked List chosen"
        def createA():
            print "Array chosen"
        def scanButton():
            print "Image processed"

            graphb = QtGui.QPushButton('Graph', self)
            graphb.setCheckable(True)
            graphb.move(10,10)

            graphb.clicked[bool].connect(createLL())

            arrayb = QtGui.QPushButton('Array',self)
            arrayb.setCheckable(True)
            arrayb.move(10,110)

            arrayb.clicked[bool].connect(createA())

            scanb=QtGui.QPushButton('Scan',self)
            scanb.move(10,210)

def main():

    app = QtGui.QApplication(sys.argv)
    ex = ButtonWidget()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
       

        
