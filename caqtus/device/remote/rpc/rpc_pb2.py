# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rpc.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\trpc.proto"\xaa\x01\n\x0b\x43\x61llRequest\x12\x10\n\x08\x66unction\x18\x01 \x01(\x0c\x12\x0c\n\x04\x61rgs\x18\x02 \x03(\x0c\x12(\n\x06kwargs\x18\x03 \x03(\x0b\x32\x18.CallRequest.KwargsEntry\x12"\n\x0creturn_value\x18\x04 \x01(\x0e\x32\x0c.ReturnValue\x1a-\n\x0bKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01">\n\x0c\x43\x61llResponse\x12\x11\n\x07success\x18\x01 \x01(\x0cH\x00\x12\x11\n\x07\x66\x61ilure\x18\x02 \x01(\x0cH\x00\x42\x08\n\x06result*(\n\x0bReturnValue\x12\x0e\n\nSERIALIZED\x10\x00\x12\t\n\x05PROXY\x10\x01\x32\x31\n\nRemoteCall\x12#\n\x04\x43\x61ll\x12\x0c.CallRequest\x1a\r.CallResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "rpc_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_CALLREQUEST_KWARGSENTRY"]._loaded_options = None
    _globals["_CALLREQUEST_KWARGSENTRY"]._serialized_options = b"8\001"
    _globals["_RETURNVALUE"]._serialized_start = 250
    _globals["_RETURNVALUE"]._serialized_end = 290
    _globals["_CALLREQUEST"]._serialized_start = 14
    _globals["_CALLREQUEST"]._serialized_end = 184
    _globals["_CALLREQUEST_KWARGSENTRY"]._serialized_start = 139
    _globals["_CALLREQUEST_KWARGSENTRY"]._serialized_end = 184
    _globals["_CALLRESPONSE"]._serialized_start = 186
    _globals["_CALLRESPONSE"]._serialized_end = 248
    _globals["_REMOTECALL"]._serialized_start = 292
    _globals["_REMOTECALL"]._serialized_end = 341
# @@protoc_insertion_point(module_scope)
