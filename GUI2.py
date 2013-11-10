import sys
from PyQt4 import QtGui, QtCore



class gui(QtGui.QWidget):
    
    def __init__(self,master=None):
        self.app = QtGui.QApplication(sys.argv)
        QtGui.QWidget.__init__(self,master)
        self.initGui()
        sys.exit(self.app.exec_())
    
    def initGui(self):
        hbox=QtGui.QHBoxLayout(self)

        topleft = QtGui.QFrame(self)
        topleft.setLineWidth(20)
        topleft.setFrameShape(QtGui.QFrame.StyledPanel)


        topright = QtGui.QFrame(self)
        topright.setLineWidth(20)
        topright.setFrameShape(QtGui.QFrame.StyledPanel)

        topBottom = QtGui.QFrame(self)
        topBottom.setLineWidth(20)
        topBottom.setFrameShape(QtGui.QFrame.StyledPanel)
        topBottom.

        bottomBottom=QtGui.QFrame(self)
        bottomBottom.setLineWidth(20)
        bottomBottom.setFrameShape(QtGui.QFrame.StyledPanel)

        

        splitter2 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter2.addWidget(topleft)
        splitter2.addWidget(topright)
        splitter3 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter3.addWidget(topBottom)
        splitter3.addWidget(bottomBottom)
        splitter1 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(splitter2)
        splitter1.addWidget(splitter3)



        hbox.addWidget(splitter1)





        self.setLayout(hbox)
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        self.setGeometry(0,0,1440,900)
        self.setWindowTitle("WhiteBoarder")
        self.show()
        

gui()

