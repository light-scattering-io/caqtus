# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\resources\sequence_execute_shot.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ExecuteShot(object):
    def setupUi(self, ExecuteShot):
        ExecuteShot.setObjectName("ExecuteShot")
        ExecuteShot.resize(400, 300)
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        ExecuteShot.setFont(font)
        self.horizontalLayout = QtWidgets.QHBoxLayout(ExecuteShot)
        self.horizontalLayout.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(ExecuteShot)
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        self.label.setFont(font)
        self.label.setStyleSheet("color: #CC7832")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.name_edit = QtWidgets.QLineEdit(ExecuteShot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_edit.sizePolicy().hasHeightForWidth())
        self.name_edit.setSizePolicy(sizePolicy)
        self.name_edit.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        self.name_edit.setFont(font)
        self.name_edit.setAutoFillBackground(False)
        self.name_edit.setStyleSheet(":focus{color: #FFC66D}\n"
":!focus{ border: none; color: #FFC66D }\n"
"\n"
"")
        self.name_edit.setText("")
        self.name_edit.setFrame(True)
        self.name_edit.setDragEnabled(False)
        self.name_edit.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.name_edit.setClearButtonEnabled(False)
        self.name_edit.setObjectName("name_edit")
        self.horizontalLayout.addWidget(self.name_edit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalLayout.setStretch(2, 1)

        self.retranslateUi(ExecuteShot)
        QtCore.QMetaObject.connectSlotsByName(ExecuteShot)

    def retranslateUi(self, ExecuteShot):
        _translate = QtCore.QCoreApplication.translate
        ExecuteShot.setWindowTitle(_translate("ExecuteShot", "Form"))
        self.label.setText(_translate("ExecuteShot", "Do"))
        self.name_edit.setPlaceholderText(_translate("ExecuteShot", "variable name"))
