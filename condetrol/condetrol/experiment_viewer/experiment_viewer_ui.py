# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\resources\experiment_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(848, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/caqtus-logo"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 848, 22))
        self.menubar.setObjectName("menubar")
        self.menuPreferences = QtWidgets.QMenu(self.menubar)
        self.menuPreferences.setObjectName("menuPreferences")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.sequences_widget = QtWidgets.QDockWidget(MainWindow)
        self.sequences_widget.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.sequences_widget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        self.sequences_widget.setObjectName("sequences_widget")
        self.dockWidgetContents_7 = QtWidgets.QWidget()
        self.dockWidgetContents_7.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.dockWidgetContents_7.setAutoFillBackground(False)
        self.dockWidgetContents_7.setObjectName("dockWidgetContents_7")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents_7)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.sequences_view = QtWidgets.QTreeView(self.dockWidgetContents_7)
        self.sequences_view.setObjectName("sequences_view")
        self.verticalLayout.addWidget(self.sequences_view)
        self.sequences_widget.setWidget(self.dockWidgetContents_7)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.sequences_widget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_edit_config = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/gear--pencil"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_edit_config.setIcon(icon1)
        self.action_edit_config.setObjectName("action_edit_config")
        self.action_start = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/control"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_start.setIcon(icon2)
        self.action_start.setObjectName("action_start")
        self.action_stop = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/control-stop"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_stop.setIcon(icon3)
        self.action_stop.setObjectName("action_stop")
        self.menuPreferences.addAction(self.action_edit_config)
        self.menubar.addAction(self.menuPreferences.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Caqtus"))
        self.menuPreferences.setTitle(_translate("MainWindow", "Edit"))
        self.sequences_widget.setWindowTitle(_translate("MainWindow", "Sequences"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_edit_config.setText(_translate("MainWindow", "Config..."))
        self.action_start.setText(_translate("MainWindow", "Start"))
        self.action_stop.setText(_translate("MainWindow", "Stop"))
import resources_rc
