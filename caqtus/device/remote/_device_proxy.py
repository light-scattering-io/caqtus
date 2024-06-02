import contextlib
from collections.abc import Callable
from typing import Self, ParamSpec, TypeVar, Generic, LiteralString, Any

from .rpc import Client, Proxy
from .. import Device

P = ParamSpec("P")

DeviceType = TypeVar("DeviceType", bound=Device)


class DeviceProxy(Generic[DeviceType]):
    def __init__(
        self,
        rpc_client: Client,
        device_type: Callable[P, DeviceType],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> None:
        self._rpc_client = rpc_client
        self._device_type = device_type
        self._args = args
        self._kwargs = kwargs
        self._device_proxy: Proxy[DeviceType]

        self._async_exit_stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> Self:
        self._device_proxy = await self._async_exit_stack.enter_async_context(
            self._rpc_client.call_proxy_result(
                self._device_type, *self._args, **self._kwargs
            )
        )
        await self._async_exit_stack.enter_async_context(
            self._rpc_client.async_context_manager(self._device_proxy)
        )
        return self

    async def get_attribute(self, attribute_name: LiteralString) -> Any:
        return await self._rpc_client.get_attribute(self._device_proxy, attribute_name)

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._async_exit_stack.__aexit__(exc_type, exc_value, traceback)