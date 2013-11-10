from PyQt4 import QtGui


class OutLog:
    def __init__(self, edit, out=None, color=None):
        self.edit = edit
        self.out = out
        self.color = color

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
