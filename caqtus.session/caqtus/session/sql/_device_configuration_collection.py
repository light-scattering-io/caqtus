from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import attrs
import sqlalchemy
from sqlalchemy.orm import Mapped, mapped_column, Session

from ._table_base import Base
from .device_configuration_serializer import DeviceConfigurationSerializer
from ..device_configuration_collection import DeviceConfigurationCollection

if TYPE_CHECKING:
    from ._experiment_session import SQLExperimentSession


class SQLDefaultDeviceConfiguration(Base):
    __tablename__ = "default_device_configurations"

    name: Mapped[str] = mapped_column(primary_key=True)
    device_type: Mapped[str] = mapped_column()
    content = mapped_column(sqlalchemy.types.JSON)


@attrs.frozen
class SQLDeviceConfigurationCollection(DeviceConfigurationCollection):
    parent_session: "SQLExperimentSession"
    device_configuration_serializers: Mapping[str, DeviceConfigurationSerializer]

    def __setitem__(self, __key, __value):
        type_name = type(__value).__qualname__
        serializer = self.device_configuration_serializers[type_name]
        content = serializer.dumper(__value)
        if __key in self:
            stmt = (
                sqlalchemy.update(SQLDefaultDeviceConfiguration)
                .where(SQLDefaultDeviceConfiguration.name == str(__key))
                .values(content=content)
            )
        else:
            stmt = sqlalchemy.insert(SQLDefaultDeviceConfiguration).values(
                name=str(__key), device_type=type_name, content=content
            )
        self._get_sql_session().execute(stmt)

    def __delitem__(self, __key):
        if __key not in self:
            raise KeyError(__key)
        stmt = sqlalchemy.delete(SQLDefaultDeviceConfiguration).where(
            SQLDefaultDeviceConfiguration.name == str(__key)
        )
        self._get_sql_session().execute(stmt)

    def __getitem__(self, __key):
        stmt = sqlalchemy.select(SQLDefaultDeviceConfiguration).where(
            SQLDefaultDeviceConfiguration.name == str(__key)
        )
        result = self._get_sql_session().execute(stmt)
        if found := result.scalar():
            serializer = self.device_configuration_serializers[found.device_type]
            device_config = serializer.loader(found.content)
            return device_config
        else:
            raise KeyError(__key)

    def __len__(self):
        stmt = sqlalchemy.select(sqlalchemy.func.count(SQLDefaultDeviceConfiguration))
        result = self._get_sql_session().execute(stmt)
        return result.scalar()

    def __iter__(self):
        stmt = sqlalchemy.select(SQLDefaultDeviceConfiguration).order_by(
            SQLDefaultDeviceConfiguration.name
        )
        result = self._get_sql_session().execute(stmt)
        return (row.name for row in result.scalars())

    def __contains__(self, item):
        stmt = sqlalchemy.select(
            sqlalchemy.func.count(SQLDefaultDeviceConfiguration.name)
        ).where(SQLDefaultDeviceConfiguration.name == str(item))
        result = self._get_sql_session().execute(stmt)
        return result.scalar() > 0

    def _get_sql_session(self) -> Session:
        # noinspection PyProtectedMember
        return self.parent_session._get_sql_session()