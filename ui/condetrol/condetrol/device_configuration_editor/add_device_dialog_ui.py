# Form implementation generated from reading ui file '.\add_device_dialog.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_AddDeviceDialog(object):
    def setupUi(self, AddDeviceDialog):
        AddDeviceDialog.setObjectName("AddDeviceDialog")
        AddDeviceDialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(AddDeviceDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.deviceNameLabel = QtWidgets.QLabel(parent=AddDeviceDialog)
        self.deviceNameLabel.setObjectName("deviceNameLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.deviceNameLabel)
        self.device_name_line_edit = QtWidgets.QLineEdit(parent=AddDeviceDialog)
        self.device_name_line_edit.setObjectName("device_name_line_edit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.device_name_line_edit)
        self.deviceTypeLabel = QtWidgets.QLabel(parent=AddDeviceDialog)
        self.deviceTypeLabel.setObjectName("deviceTypeLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.deviceTypeLabel)
        self.device_type_combo_box = QtWidgets.QComboBox(parent=AddDeviceDialog)
        self.device_type_combo_box.setObjectName("device_type_combo_box")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.device_type_combo_box)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=AddDeviceDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(AddDeviceDialog)
        self.buttonBox.accepted.connect(AddDeviceDialog.accept) # type: ignore
        self.buttonBox.rejected.connect(AddDeviceDialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(AddDeviceDialog)

    def retranslateUi(self, AddDeviceDialog):
        _translate = QtCore.QCoreApplication.translate
        AddDeviceDialog.setWindowTitle(_translate("AddDeviceDialog", "Add device..."))
        self.deviceNameLabel.setText(_translate("AddDeviceDialog", "Device name"))
        self.deviceTypeLabel.setText(_translate("AddDeviceDialog", "Device type"))
