import sys

from PyQt6.QtWidgets import QApplication

from analyza.loading.importers import ImageImporter
from device.configuration import DeviceName
from sequence.runtime import Sequence
from viewer.sequence_watcher import SequenceWatcher
from viewer.single_shot_viewers import SingleShotWidget, ImageViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = SingleShotWidget(
        viewers=[
            ImageViewer(importer=ImageImporter(DeviceName("Orca Quest"), "picture")),
            ImageViewer(importer=ImageImporter(DeviceName("Orca Quest"), "picture 2")),
        ]
    )
    widget.show()

    watcher = SequenceWatcher(
        Sequence("2023.06_June.14.test"),
        target=widget.add_shots,
        watch_interval=0.1,
    )
    watcher.start()

    app.exec()
