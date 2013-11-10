from PyQt4 import QtGui
import sys


class OutLog(QtGui.QWidget):
    def __init__(self, edit, out=sys.stdout, color=None,master=None):
        QtGui.QWidget.__init__(self,master)
        self.app=QtGui.QApplication(sys.argv)
        self.edit = edit
        self.out = out
        self.color = color
        sys.exit(self.app.exec_())

    def write(self, message):
        if self.color:
            col = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText(message)

        if self.color:
            self.edit.setTextColor(col)

        if self.out:
            self.out.write(message)

#import sys
#sys.stdout = OutLog( edit, sys.stdout)
#sys.stderr = OutLog( edit, sys.stderr, QtGui.QColor(255,0,0) )


q = QtGui.QPlainTextEdit()
o = OutLog(q)
