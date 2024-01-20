from __future__ import annotations

import multiprocessing.managers
import time
import uuid
from collections.abc import Set
from typing import Optional

from core.compilation import ShotCompilerFactory
from core.session import ExperimentSessionMaker
from core.session import PureSequencePath

from .manager import ExperimentManager, Procedure, BoundExperimentManager
from ..sequence_runner import ShotRetryConfig
from ..shot_runner import ShotRunnerFactory

experiment_manager: Optional[BoundExperimentManager] = None


class ExperimentManagerProxy(ExperimentManager, multiprocessing.managers.BaseProxy):
    _exposed_ = "create_procedure"
    _method_to_typeid_ = {
        "create_procedure": "ProcedureProxy",
    }

    def create_procedure(
        self, procedure_name: str, acquisition_timeout: Optional[float] = None
    ) -> ProcedureProxy:
        return self._callmethod("create_procedure", (procedure_name,))  # type: ignore

    def __repr__(self):
        return f"<ExperimentManagerProxy at {hex(id(self))}>"


class ProcedureProxy(Procedure, multiprocessing.managers.BaseProxy):
    """Proxy for a procedure running in a different process.

    This object behaves like a :class:`Procedure` object and should be used like it, but
    it forwards all method calls to an actual procedure object running in a different
    process.
    """

    _exposed_ = (
        "__enter__",
        "__exit__",
        "is_active",
        "is_running_sequence",
        "sequences",
        "exception",
        "start_sequence",
        "run_sequence",
        "__str__",
    )
    _method_to_typeid_ = {"__enter__": "ProcedureProxy"}

    def __enter__(self):
        return self._callmethod("__enter__", ())

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._callmethod("__exit__", (None, exc_val, None))

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self}') at {hex(id(self))}>"

    def __str__(self):
        return self._callmethod("__str__", ())

    def is_active(self) -> bool:
        return self._callmethod("is_active", ())

    def is_running_sequence(self) -> bool:
        return self._callmethod("is_running_sequence", ())

    def sequences(self) -> list[PureSequencePath]:
        return self._callmethod("sequences", ())

    def exception(self) -> Optional[Exception]:
        return self._callmethod("exception", ())

    def start_sequence(
        self,
        sequence_path: PureSequencePath,
        device_configurations_uuids: Optional[Set[uuid.UUID]] = None,
        constant_tables_uuids: Optional[Set[uuid.UUID]] = None,
    ) -> None:
        return self._callmethod("start_sequence", (sequence_path,))


class _MultiprocessingServerManager(multiprocessing.managers.BaseManager):
    pass


def _get_experiment_manager() -> BoundExperimentManager:
    if experiment_manager is None:
        raise RuntimeError("Experiment manager not initialized")
    return experiment_manager


def _enter_experiment_manager() -> None:
    experiment_manager.__enter__()


def _exit_experiment_manager(exc_value) -> None:
    experiment_manager.__exit__(type(exc_value), exc_value, exc_value.__traceback__)


def _create_experiment_manager(
    session_maker: ExperimentSessionMaker,
    shot_compiler_factory: ShotCompilerFactory,
    shot_runner_factory: ShotRunnerFactory,
    shot_retry_config: Optional[ShotRetryConfig] = None,
) -> None:
    global experiment_manager
    experiment_manager = BoundExperimentManager(
        session_maker, shot_compiler_factory, shot_runner_factory, shot_retry_config
    )


_MultiprocessingServerManager.register(
    "create_experiment_manager", _create_experiment_manager, ExperimentManagerProxy
)
_MultiprocessingServerManager.register(
    "get_experiment_manager", _get_experiment_manager, ExperimentManagerProxy
)
_MultiprocessingServerManager.register(
    "enter_experiment_manager", _enter_experiment_manager
)
_MultiprocessingServerManager.register(
    "exit_experiment_manager", _exit_experiment_manager
)
_MultiprocessingServerManager.register("ProcedureProxy", None, ProcedureProxy)


class RemoteExperimentManagerServer:
    session_maker: Optional[ExperimentSessionMaker] = None

    def __init__(
        self,
        address: tuple[str, int],
        authkey: bytes,
        session_maker: ExperimentSessionMaker,
        shot_compiler_factory: ShotCompilerFactory,
        shot_runner_factory: ShotRunnerFactory,
        shot_retry_config: Optional[ShotRetryConfig] = None,
    ):
        self._session_maker = session_maker
        self._multiprocessing_manager = _MultiprocessingServerManager(
            address=address, authkey=authkey
        )
        self._shot_compiler_factory = shot_compiler_factory
        self._shot_runner_factory = shot_runner_factory
        self._shot_retry_config = shot_retry_config
        self._shot_runner_factory = shot_runner_factory
        self._shot_retry_config = shot_retry_config

    def __enter__(self):
        self._multiprocessing_manager.start()
        self._multiprocessing_manager.create_experiment_manager(
            self._session_maker,
            self._shot_compiler_factory,
            self._shot_runner_factory,
            self._shot_retry_config,
        )
        self._multiprocessing_manager.enter_experiment_manager()
        return self

    @staticmethod
    def serve_forever():
        while True:
            time.sleep(100e-3)

    def __exit__(self, exc_type, exc_value, traceback):
        self._multiprocessing_manager.exit_experiment_manager(exc_value)
        return self._multiprocessing_manager.__exit__(exc_type, exc_value, traceback)


class _MultiprocessingClientManager(multiprocessing.managers.BaseManager):
    pass


_MultiprocessingClientManager.register(
    "get_experiment_manager", None, ExperimentManagerProxy
)

_MultiprocessingClientManager.register("ProcedureProxy", None, ProcedureProxy)


class RemoteExperimentManagerClient:
    def __init__(self, address: tuple[str, int], authkey: bytes):
        self._multiprocessing_manager = _MultiprocessingClientManager(
            address=address, authkey=authkey
        )
        self._multiprocessing_manager.connect()

    def get_experiment_manager(self) -> ExperimentManagerProxy:
        return self._multiprocessing_manager.get_experiment_manager()
