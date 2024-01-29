from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
from core.device import DeviceName, DeviceConfigurationAttrs

from .iteration_configuration import IterationConfiguration
from .shot import Shot
from .._return_or_raise import unwrap
from ..path import PureSequencePath
from ..shot import TimeLanes

if TYPE_CHECKING:
    from ..experiment_session import ExperimentSession


@attrs.frozen(eq=False, order=False)
class Sequence:
    """Contains the runtime information and data of a sequence.

    Only methods that take an ExperimentSession argument actually connect to the
    permanent storage of the experiment. Such methods can raise SequenceNotFoundError if
    the sequence does not exist in the session. They are also expected to be
    comparatively slow since they require a file system access, possibly over the
    network.
    """

    path: PureSequencePath

    def __str__(self) -> str:
        return str(self.path)

    def exists(self, session: ExperimentSession) -> bool:
        """Check if the sequence exists in the session."""

        if session.paths.does_path_exists(self.path):
            return unwrap(session.sequences.is_sequence(self.path))
        else:
            return False

    def get_iteration_configuration(
        self, session: ExperimentSession
    ) -> IterationConfiguration:
        """Return the iteration configuration of the sequence."""

        return session.sequences.get_iteration_configuration(self.path)

    def get_time_lanes(self, session: ExperimentSession) -> TimeLanes:
        """Return the time lanes that define how a shot is run for this sequence."""

        return session.sequences.get_time_lanes(self.path)

    def set_time_lanes(self, time_lanes: TimeLanes, session: ExperimentSession) -> None:
        """Set the time lanes that define how a shot is run for this sequence."""

        return session.sequences.set_time_lanes(self.path, time_lanes)

    def get_shots(self, session: ExperimentSession) -> list[Shot]:
        """Return the shots that belong to this sequence."""

        return session.sequences.get_shots(self.path)

    def duplicate(
        self, target_path: PureSequencePath | str, session: ExperimentSession
    ) -> Sequence:
        """Duplicate the sequence to a new path.

        The sequence created will be in the draft state and will have the same iteration
        configuration and time lanes as the original sequence.
        """

        if isinstance(target_path, str):
            target_path = PureSequencePath(target_path)

        iteration_configuration = self.get_iteration_configuration(session)
        time_lanes = self.get_time_lanes(session)
        return session.sequences.create(
            target_path, iteration_configuration, time_lanes
        )

    def get_device_configurations(
        self, session: ExperimentSession
    ) -> dict[DeviceName, DeviceConfigurationAttrs]:
        """Return the device configurations used when the sequence was launched."""

        device_uuids = session.sequences.get_device_configuration_uuids(self.path)

        return {
            session.device_configurations.get_device_name(
                device_uuid
            ): session.device_configurations.get_configuration(device_uuid)
            for device_uuid in device_uuids
        }

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return self.path == other.path
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.path)
