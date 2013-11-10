import sys
from PyQt4 import QtGui
from PyQt4 import QtCore

from Console import EmittingStream

class gui(QtGui.QWidget):
    
    def __init__(self,master=None):

        app = QtGui.QApplication(sys.argv)
        QtGui.QWidget.__init__(self,master)
        self.textEdit = QtGui.QTextEdit()
        self.initGui()

        self.stream =  EmittingStream(textWritten=self.normalOutputWritten)
        sys.stdout=self.stream
        
        sys.exit(app.exec_())       

    def __del__(self):
        sys.stdout = sys.__stdout__
        
    def normalOutputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()
    
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

        bottomBottom=QtGui.QFrame(self)
        bottomBottom.setLineWidth(20)
        bottomBottom.setFrameShape(QtGui.QFrame.StyledPanel)
        textLayout = QtGui.QHBoxLayout()
        textLayout.addWidget(self.textEdit)
        bottomBottom.setLayout(textLayout)

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



