# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: vlm_chat.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'vlm_chat.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0evlm_chat.proto\x12\x10vlm_chat_service\"1\n\x0b\x43hatRequest\x12\x0e\n\x06imdata\x18\x01 \x03(\x0c\x12\x12\n\nprompt_str\x18\x02 \x01(\t\"=\n\x03\x42ox\x12\x0c\n\x04xmin\x18\x01 \x01(\x05\x12\x0c\n\x04ymin\x18\x02 \x01(\x05\x12\x0c\n\x04xmax\x18\x03 \x01(\x05\x12\x0c\n\x04ymax\x18\x04 \x01(\x05\"$\n\x0c\x43hatResponse\x12\x14\n\x0cresponse_str\x18\x01 \x01(\t\"%\n\x06Tensor\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x02\x12\r\n\x05shape\x18\x02 \x03(\x05\x32_\n\x0eVLMChatService\x12M\n\nVLMOneChat\x12\x1d.vlm_chat_service.ChatRequest\x1a\x1e.vlm_chat_service.ChatResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vlm_chat_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CHATREQUEST']._serialized_start=36
  _globals['_CHATREQUEST']._serialized_end=85
  _globals['_BOX']._serialized_start=87
  _globals['_BOX']._serialized_end=148
  _globals['_CHATRESPONSE']._serialized_start=150
  _globals['_CHATRESPONSE']._serialized_end=186
  _globals['_TENSOR']._serialized_start=188
  _globals['_TENSOR']._serialized_end=225
  _globals['_VLMCHATSERVICE']._serialized_start=227
  _globals['_VLMCHATSERVICE']._serialized_end=322
# @@protoc_insertion_point(module_scope)