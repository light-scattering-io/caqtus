# Form implementation generated from reading ui file '.\main_window.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_CondetrolMainWindow(object):
    def setupUi(self, CondetrolMainWindow):
        CondetrolMainWindow.setObjectName("CondetrolMainWindow")
        CondetrolMainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(parent=CondetrolMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        CondetrolMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=CondetrolMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.device_configurations_menu = QtWidgets.QMenu(parent=self.menubar)
        self.device_configurations_menu.setObjectName("device_configurations_menu")
        self.menu = QtWidgets.QMenu(parent=self.menubar)
        self.menu.setObjectName("menu")
        self.menuConstants = QtWidgets.QMenu(parent=self.menubar)
        self.menuConstants.setObjectName("menuConstants")
        CondetrolMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=CondetrolMainWindow)
        self.statusbar.setObjectName("statusbar")
        CondetrolMainWindow.setStatusBar(self.statusbar)
        self.action_edit_device_configurations = QtGui.QAction(parent=CondetrolMainWindow)
        self.action_edit_device_configurations.setObjectName("action_edit_device_configurations")
        self.actionExport = QtGui.QAction(parent=CondetrolMainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionLoad = QtGui.QAction(parent=CondetrolMainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.action_edit_constants = QtGui.QAction(parent=CondetrolMainWindow)
        self.action_edit_constants.setObjectName("action_edit_constants")
        self.device_configurations_menu.addAction(self.action_edit_device_configurations)
        self.menuConstants.addAction(self.action_edit_constants)
        self.menubar.addAction(self.menuConstants.menuAction())
        self.menubar.addAction(self.device_configurations_menu.menuAction())
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(CondetrolMainWindow)
        QtCore.QMetaObject.connectSlotsByName(CondetrolMainWindow)

    def retranslateUi(self, CondetrolMainWindow):
        _translate = QtCore.QCoreApplication.translate
        CondetrolMainWindow.setWindowTitle(_translate("CondetrolMainWindow", "Condetrol"))
        self.device_configurations_menu.setTitle(_translate("CondetrolMainWindow", "Devices"))
        self.menu.setTitle(_translate("CondetrolMainWindow", "Remote servers"))
        self.menuConstants.setTitle(_translate("CondetrolMainWindow", "Constants"))
        self.action_edit_device_configurations.setText(_translate("CondetrolMainWindow", "Edit..."))
        self.actionExport.setText(_translate("CondetrolMainWindow", "Export..."))
        self.actionLoad.setText(_translate("CondetrolMainWindow", "Load..."))
        self.action_edit_constants.setText(_translate("CondetrolMainWindow", "Edit..."))
