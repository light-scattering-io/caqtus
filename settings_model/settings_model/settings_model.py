import abc
import typing as th

import pydantic
import yaml


class SettingsModel(abc.ABC, pydantic.BaseModel):
    """Allows to store and load experiment configuration with type validation

    All instances of a subclass of this class can be (de)serialized (from) to yaml based on their fields (see pydantic).
    This is used for persistence and loading of experiment settings. If you need to add some parameters to the
    experiment manager, it is recommended to create a new subclass of this class.
    """

    class Config:
        validate_assignment = True

    def __init_subclass__(cls):
        """Register subclasses for serialization and deserialization to yaml"""
        yaml.SafeDumper.add_representer(cls, generate_representer(cls))
        yaml.SafeLoader.add_constructor(f"!{cls.__name__}", generate_constructor(cls))


def generate_representer(class_: th.Type[SettingsModel]):
    def representer(dumper: yaml.Dumper, settings: SettingsModel):
        return dumper.represent_mapping(
            f"!{class_.__name__}",
            {field: getattr(settings, field) for field in class_.__fields__},
        )

    return representer


def generate_constructor(class_: th.Type[SettingsModel]):
    def constructor(loader: yaml.Loader, node: yaml.Node):
        return class_(**loader.construct_mapping(node, deep=True))

    return constructor
