from datetime import datetime
from typing import (
    Any,
    TypeVar,
    ParamSpec,
    TypedDict,
)

from experiment.session import ExperimentSession
from sequence.runtime import Shot
from .chainable_function import ChainableFunction

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")

K = TypeVar("K")
V = TypeVar("V")


def _import_parameters(shot: Shot, session: ExperimentSession) -> dict[str, Any]:
    return shot.get_parameters(session)


def _import_scores(shot: Shot, session: ExperimentSession) -> dict[str, Any]:
    scores = shot.get_scores(session)
    return {f"{key}.score": value for key, value in scores.items()}


def _import_measures(shot: Shot, session: ExperimentSession) -> dict[str, Any]:
    result = {}
    data = shot.get_measures(session)
    for device, device_date in data.items():
        for key, value in device_date.items():
            result[f"{device}.{key}"] = value
    return result


class _ImportTimeResult(TypedDict):
    start_time: datetime
    end_time: datetime


def _import_time(shot: Shot, session: ExperimentSession) -> _ImportTimeResult:
    return _ImportTimeResult(
        start_time=shot.get_start_time(session), end_time=shot.get_end_time(session)
    )


def _import_all(shot: Shot, session: ExperimentSession) -> dict[str, Any]:
    return dict(
        **_import_time(shot, session),
        **_import_scores(shot, session),
        **_import_parameters(shot, session),
        **_import_measures(shot, session),
    )


import_all = ChainableFunction(_import_all)
import_parameters = ChainableFunction(_import_parameters)
import_scores = ChainableFunction(_import_scores)
import_measures = ChainableFunction(_import_measures)
import_time = ChainableFunction(_import_time)
