# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\condetrol\experiment_viewer\resources\sequence_variable_declaration.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_VariableDeclaration(object):
    def setupUi(self, VariableDeclaration):
        VariableDeclaration.setObjectName("VariableDeclaration")
        VariableDeclaration.resize(904, 56)
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        font.setBold(True)
        font.setWeight(75)
        VariableDeclaration.setFont(font)
        VariableDeclaration.setMouseTracking(True)
        VariableDeclaration.setAutoFillBackground(False)
        self.horizontalLayout = QtWidgets.QHBoxLayout(VariableDeclaration)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.name_edit = AutoResizeLineEdit(VariableDeclaration)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_edit.sizePolicy().hasHeightForWidth())
        self.name_edit.setSizePolicy(sizePolicy)
        self.name_edit.setMinimumSize(QtCore.QSize(300, 0))
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        font.setBold(False)
        font.setWeight(50)
        self.name_edit.setFont(font)
        self.name_edit.setAutoFillBackground(False)
        self.name_edit.setStyleSheet(":focus{color: #AA4926}\n"
":!focus{ border: none; color: #AA4926 }\n"
"\n"
"")
        self.name_edit.setText("")
        self.name_edit.setFrame(True)
        self.name_edit.setDragEnabled(False)
        self.name_edit.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.name_edit.setClearButtonEnabled(False)
        self.name_edit.setObjectName("name_edit")
        self.horizontalLayout.addWidget(self.name_edit)
        self.label_2 = QtWidgets.QLabel(VariableDeclaration)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setStyleSheet("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.expression_edit = AutoResizeLineEdit(VariableDeclaration)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.expression_edit.sizePolicy().hasHeightForWidth())
        self.expression_edit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("JetBrains Mono")
        font.setBold(False)
        font.setWeight(50)
        self.expression_edit.setFont(font)
        self.expression_edit.setAutoFillBackground(False)
        self.expression_edit.setStyleSheet(":focus{ color: #6897BB }\n"
":!focus{ border: none; color: #6897BB }")
        self.expression_edit.setObjectName("expression_edit")
        self.horizontalLayout.addWidget(self.expression_edit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalLayout.setStretch(3, 1)

        self.retranslateUi(VariableDeclaration)
        QtCore.QMetaObject.connectSlotsByName(VariableDeclaration)

    def retranslateUi(self, VariableDeclaration):
        _translate = QtCore.QCoreApplication.translate
        VariableDeclaration.setWindowTitle(_translate("VariableDeclaration", "Form"))
        self.name_edit.setPlaceholderText(_translate("VariableDeclaration", "variable name"))
        self.label_2.setText(_translate("VariableDeclaration", "="))
        self.expression_edit.setPlaceholderText(_translate("VariableDeclaration", "expression"))
from widgets.auto_resize_lineedit import AutoResizeLineEdit
