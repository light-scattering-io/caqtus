import contextlib
from collections.abc import AsyncIterable, AsyncGenerator

from caqtus.device.controller import (
    DeviceController,
    run_in_thread,
    async_context,
    iterate_async,
)
from caqtus.types.data import DataLabel
from caqtus.types.image import Image
from ._runtime import Camera


class CameraController(DeviceController):
    async def run_shot(
        self,
        camera: Camera,
        /,
        timeout: float,
        picture_names: list[str],
        exposures: list[float],
    ) -> None:
        await self.set_timeout(camera, timeout)
        async with self.acquire_pictures(camera, picture_names, exposures) as pictures:
            async for picture in pictures:
                pass

    @staticmethod
    async def set_timeout(camera: Camera, timeout: float) -> None:
        await run_in_thread(camera.update_parameters, timeout=timeout)

    @contextlib.asynccontextmanager
    async def acquire_pictures(
        self, camera: Camera, picture_names: list[str], exposures: list[float]
    ) -> AsyncGenerator[AsyncIterable[tuple[str, Image]], None]:
        if len(picture_names) != len(exposures):
            raise ValueError(
                f"Number of picture names ({len(picture_names)}) must be equal to the "
                f"number of exposures ({len(exposures)})"
            )
        async with async_context(camera.acquire(exposures)) as pictures:
            self.signal_ready()
            yield self.emit_signals(
                iterate_async(zip(picture_names, pictures, strict=True))
            )

    async def emit_signals(
        self, named_pictures: AsyncIterable[tuple[str, Image]]
    ) -> AsyncGenerator[Image, None]:
        async for name, picture in named_pictures:
            self.signal_data_acquired(DataLabel(rf"{self.device_name}\{name}"), picture)
            yield picture