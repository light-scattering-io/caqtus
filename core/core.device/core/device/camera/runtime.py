import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import ClassVar, Optional

import numpy
from attrs import define, field
from attrs.setters import frozen, validate, convert, pipe
from attrs.validators import instance_of, deep_iterable

from core.types.image import Image, ImageLabel
from util import log_exception
from .configuration import RectangularROI
from ..runtime import RuntimeDevice

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class CameraTimeoutError(TimeoutError):
    pass


@define
class Camera(RuntimeDevice, ABC):
    """Define the interface for a camera.

    This is an abstract class that must be subclassed to implement a specific camera.
    When using a device inheriting from this class , it is required to know the
    number of pictures that will be acquired before starting an acquisition. Devices
    of this class are not meant to be used in video mode.

    Attributes:
        picture_names: Names to give to the pictures in order of acquisition. Each name
            must be unique. This will define the number of picture to take during one
            acquisition, and it is frozen after initialization.
        roi: The region of interest to keep from the full sensor image. Depending on the
            device, this can be enforced before or after retrieving the image from the
            camera.
        timeout: The camera must raise a CameraTimeoutError if it didn't receive a
            trigger within this time after starting acquisition.
            The timeout is in seconds.
        exposures: List of exposures to use for the pictures to acquire. The length of
            the list must match the length of picture_names.
            Each exposure is in seconds.
        external_trigger: Specify if the camera should wait for an external trigger to
            take a picture. If set to False, it should acquire images as fast as
            possible.

    Classes inheriting of CCamera must implement the following methods:
    - _start_acquisition
    - _stop_acquisition
    - _is_acquisition_in_progress
    """

    sensor_width: ClassVar[int]
    sensor_height: ClassVar[int]

    picture_names: tuple[ImageLabel, ...] = field(
        converter=tuple,
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(tuple)
        ),
        on_setattr=frozen,
    )
    roi: RectangularROI = field(
        validator=instance_of(RectangularROI), on_setattr=frozen
    )
    timeout: float = field(converter=float, on_setattr=convert)
    exposures: list[float] = field(
        converter=list,
        validator=deep_iterable(
            member_validator=instance_of(float), iterable_validator=instance_of(list)
        ),
        on_setattr=pipe(convert, validate),
    )
    external_trigger: bool = field(validator=instance_of(bool), on_setattr=frozen)

    _pictures: list[Optional[numpy.ndarray]] = field(factory=list, init=False)

    @picture_names.validator  # type: ignore
    def validate_picture_names(self, _, picture_names):
        names = list(picture_names)
        counts = Counter(names)
        for name, count in counts.items():
            if count > 1:
                raise ValueError(f"Picture name {name} is used several times")

    @exposures.validator  # type: ignore
    def validate_exposures(self, _, exposures):
        num_exposures = len(exposures)
        num_names = len(self.picture_names)
        if num_names != num_exposures:
            raise ValueError(
                f"Number of picture names ({num_names}) and of exposures"
                f" ({num_exposures}) must match"
            )

        if any(exposure > self.timeout for exposure in exposures):
            raise ValueError(f"Exposure is longer than timeout")

    @log_exception(logger)
    def update_parameters(self, exposures: list[float], timeout: float) -> None:
        """Update the exposures time of the camera"""

        if not (self.are_all_pictures_acquired() or self.no_pictures_acquired()):
            self.stop_acquisition()

        super().update_parameters(exposures=exposures, timeout=timeout)

    def are_all_pictures_acquired(self):
        return all(picture is not None for picture in self._pictures)

    def no_pictures_acquired(self):
        return all(picture is None for picture in self._pictures)

    def start_acquisition(self):
        self._pictures = [None] * self.number_pictures_to_acquire
        self._start_acquisition(self.number_pictures_to_acquire)

    def is_acquisition_in_progress(self) -> bool:
        return self._is_acquisition_in_progress()

    def stop_acquisition(self):
        self._stop_acquisition()

    @abstractmethod
    def _start_acquisition(self, number_pictures: int):
        """Start the acquisition of pictures

        To implement in subclasses.

        Actual camera implementation must implement this method. It must start the
        acquisition of pictures and return as soon as possible. It should raise an
        error if the acquisition could not be started or is already in progress. The
        acquisition must be stopped by calling _stop_acquisition.

        Args:
            number_pictures: Number of pictures to acquire.

        Raises: CameraTimeoutError: If the camera didn't receive a trigger within the
        timeout after starting acquisition. If this error is raised, the acquisition
        will be stopped, but it informs the experiment manager that it can retry this
        acquisition.
        """
        ...

    @abstractmethod
    def _is_acquisition_in_progress(self) -> bool:
        """Return True if the acquisition is in progress, False otherwise

        To implement in subclasses.
        """
        ...

    @abstractmethod
    def _stop_acquisition(self):
        """Stop the acquisition of pictures

        To implement in subclasses.
        """
        ...

    def acquire_all_pictures(self) -> None:
        """Take all the pictures specified by their names and exposures

        This function is blocking until all required pictures have been taken.
        """

        self.start_acquisition()
        while self.is_acquisition_in_progress():
            time.sleep(10e-3)
        self.stop_acquisition()

    def read_all_pictures(self) -> dict[ImageLabel, Image]:
        if not self.are_all_pictures_acquired():
            raise CameraTimeoutError(
                f"Not all pictures have been acquired for camera {self.name}"
            )
        else:
            return {
                name: self._pictures[index]
                for index, name in enumerate(self.picture_names)
            }

    def get_picture(self, picture_name: ImageLabel) -> Optional[Image]:
        """Return the picture with the given name.

        If the picture has not been acquired yet, return None.
        """

        return self._pictures[self.picture_names.index(picture_name)]

    def close(self):
        try:
            if self.is_acquisition_in_progress():
                self.stop_acquisition()
        finally:
            super().close()

    def get_picture_names(self) -> tuple[ImageLabel, ...]:
        return self.picture_names

    @classmethod
    def exposed_remote_methods(cls) -> tuple[str, ...]:
        return super().exposed_remote_methods() + (
            "start_acquisition",
            "is_acquisition_in_progress",
            "stop_acquisition",
            "acquire_picture",
            "acquire_all_pictures",
            "read_picture",
            "read_all_pictures",
            "reset_acquisition",
            "get_picture",
            "get_picture_names",
        )

    @property
    def number_pictures_to_acquire(self):
        return len(self.picture_names)
