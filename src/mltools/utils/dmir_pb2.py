# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dmir.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='dmir.proto',
  package='dmir',
  syntax='proto3',
  serialized_pb=_b('\n\ndmir.proto\x12\x04\x64mir\"\xc7\x01\n\x06Tensor\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05shape\x18\x02 \x03(\r\x12%\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x15.dmir.NumericalFormat\x12\r\n\x05value\x18\x04 \x03(\x02\x12\x14\n\x0cis_quantized\x18\x05 \x01(\x08\x12\x0f\n\x07qscheme\x18\x06 \x01(\t\x12\x0f\n\x07q_scale\x18\x07 \x01(\x02\x12\x14\n\x0cq_zero_point\x18\x08 \x01(\x05\x12\x0c\n\x04\x66ile\x18\t \x01(\t\x12\x0e\n\x06\x64\x65vice\x18\x0f \x01(\t\"\xb5\x03\n\tAttribute\x12+\n\x04kind\x18\x01 \x01(\x0e\x32\x1d.dmir.Attribute.AttributeKind\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x13\n\x0b\x66loat_value\x18\x03 \x01(\x02\x12\x15\n\rinteger_value\x18\x04 \x01(\x03\x12\x14\n\x0cstring_value\x18\x05 \x01(\t\x12\x13\n\x0btensor_name\x18\x06 \x01(\t\x12\x12\n\ngraph_name\x18\x07 \x01(\t\x12\x14\n\x0c\x66loat_values\x18\x08 \x03(\x02\x12\x16\n\x0einteger_values\x18\t \x03(\x03\x12\x15\n\rstring_values\x18\n \x03(\t\x12\x14\n\x0ctensor_names\x18\x0b \x03(\t\x12\x13\n\x0bgraph_names\x18\x0c \x03(\t\"\x91\x01\n\rAttributeKind\x12\r\n\tUNDEFINED\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\x07\n\x03INT\x10\x02\x12\n\n\x06STRING\x10\x03\x12\n\n\x06TENSOR\x10\x04\x12\t\n\x05GRAPH\x10\x05\x12\n\n\x06\x46LOATS\x10\x06\x12\x08\n\x04INTS\x10\x07\x12\x0b\n\x07STRINGS\x10\x08\x12\x0b\n\x07TENSORS\x10\t\x12\n\n\x06GRAPHS\x10\n\"e\n\nDependency\x12\x11\n\toperation\x18\x01 \x01(\t\x12\x10\n\x08\x61rgument\x18\x02 \x03(\t\x12\x0e\n\x06result\x18\x03 \x03(\t\x12\"\n\tattribute\x18\x04 \x03(\x0b\x32\x0f.dmir.Attribute\"\xec\x01\n\x05Graph\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07op_type\x18\x02 \x01(\t\x12\x1b\n\x05input\x18\x03 \x03(\x0b\x32\x0c.dmir.Tensor\x12\x1c\n\x06output\x18\x04 \x03(\x0b\x32\x0c.dmir.Tensor\x12\"\n\x0cintermediate\x18\x05 \x03(\x0b\x32\x0c.dmir.Tensor\x12\x1d\n\x08subgraph\x18\x06 \x03(\x0b\x32\x0b.dmir.Graph\x12$\n\ndependency\x18\x07 \x03(\x0b\x32\x10.dmir.Dependency\x12\x0e\n\x06\x64\x65vice\x18\x08 \x01(\t\x12\x10\n\x08metadata\x18\x0f \x01(\t*\xc2\x02\n\x0fNumericalFormat\x12\r\n\tUNDEFINED\x10\x00\x12\x0b\n\x07\x46LOAT32\x10\x01\x12\t\n\x05\x46LOAT\x10\x01\x12\x0b\n\x07\x46LOAT64\x10\x02\x12\n\n\x06\x44OUBLE\x10\x02\x12\x0b\n\x07\x46LOAT16\x10\x03\x12\x08\n\x04HALF\x10\x03\x12\x0c\n\x08\x42\x46LOAT16\x10\x04\x12\t\n\x05UINT8\x10\x05\x12\x08\n\x04INT8\x10\x06\x12\t\n\x05INT16\x10\x07\x12\t\n\x05SHORT\x10\x07\x12\t\n\x05INT32\x10\x08\x12\x07\n\x03INT\x10\x08\x12\t\n\x05INT64\x10\t\x12\x08\n\x04LONG\x10\t\x12\x08\n\x04\x42OOL\x10\n\x12\n\n\x06UINT16\x10\x0b\x12\n\n\x06UINT32\x10\x0c\x12\n\n\x06UINT64\x10\r\x12\x0f\n\x0b\x42\x46P16_64_FD\x10\x0e\x12\x10\n\x0c\x42\x46P12_128_FD\x10\x0f\x12\x0f\n\x0b\x42\x46P16_64_LD\x10\x10\x12\x10\n\x0c\x42\x46P12_128_LD\x10\x11\x1a\x02\x10\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_NUMERICALFORMAT = _descriptor.EnumDescriptor(
  name='NumericalFormat',
  full_name='dmir.NumericalFormat',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT32', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=2, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT64', index=3, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOUBLE', index=4, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT16', index=5, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HALF', index=6, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BFLOAT16', index=7, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT8', index=8, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT8', index=9, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT16', index=10, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SHORT', index=11, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=12, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT', index=13, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT64', index=14, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LONG', index=15, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BOOL', index=16, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT16', index=17, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT32', index=18, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UINT64', index=19, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BFP16_64_FD', index=20, number=14,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BFP12_128_FD', index=21, number=15,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BFP16_64_LD', index=22, number=16,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BFP12_128_LD', index=23, number=17,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=_descriptor._ParseOptions(descriptor_pb2.EnumOptions(), _b('\020\001')),
  serialized_start=1005,
  serialized_end=1327,
)
_sym_db.RegisterEnumDescriptor(_NUMERICALFORMAT)

NumericalFormat = enum_type_wrapper.EnumTypeWrapper(_NUMERICALFORMAT)
UNDEFINED = 0
FLOAT32 = 1
FLOAT = 1
FLOAT64 = 2
DOUBLE = 2
FLOAT16 = 3
HALF = 3
BFLOAT16 = 4
UINT8 = 5
INT8 = 6
INT16 = 7
SHORT = 7
INT32 = 8
INT = 8
INT64 = 9
LONG = 9
BOOL = 10
UINT16 = 11
UINT32 = 12
UINT64 = 13
BFP16_64_FD = 14
BFP12_128_FD = 15
BFP16_64_LD = 16
BFP12_128_LD = 17


_ATTRIBUTE_ATTRIBUTEKIND = _descriptor.EnumDescriptor(
  name='AttributeKind',
  full_name='dmir.Attribute.AttributeKind',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TENSOR', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPH', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOATS', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INTS', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRINGS', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TENSORS', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPHS', index=10, number=10,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=515,
  serialized_end=660,
)
_sym_db.RegisterEnumDescriptor(_ATTRIBUTE_ATTRIBUTEKIND)


_TENSOR = _descriptor.Descriptor(
  name='Tensor',
  full_name='dmir.Tensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='dmir.Tensor.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='shape', full_name='dmir.Tensor.shape', index=1,
      number=2, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='format', full_name='dmir.Tensor.format', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='dmir.Tensor.value', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='is_quantized', full_name='dmir.Tensor.is_quantized', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='qscheme', full_name='dmir.Tensor.qscheme', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='q_scale', full_name='dmir.Tensor.q_scale', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='q_zero_point', full_name='dmir.Tensor.q_zero_point', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='file', full_name='dmir.Tensor.file', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='device', full_name='dmir.Tensor.device', index=9,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=220,
)


_ATTRIBUTE = _descriptor.Descriptor(
  name='Attribute',
  full_name='dmir.Attribute',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='kind', full_name='dmir.Attribute.kind', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='dmir.Attribute.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_value', full_name='dmir.Attribute.float_value', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='integer_value', full_name='dmir.Attribute.integer_value', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='string_value', full_name='dmir.Attribute.string_value', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tensor_name', full_name='dmir.Attribute.tensor_name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='graph_name', full_name='dmir.Attribute.graph_name', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_values', full_name='dmir.Attribute.float_values', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='integer_values', full_name='dmir.Attribute.integer_values', index=8,
      number=9, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='string_values', full_name='dmir.Attribute.string_values', index=9,
      number=10, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tensor_names', full_name='dmir.Attribute.tensor_names', index=10,
      number=11, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='graph_names', full_name='dmir.Attribute.graph_names', index=11,
      number=12, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ATTRIBUTE_ATTRIBUTEKIND,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=223,
  serialized_end=660,
)


_DEPENDENCY = _descriptor.Descriptor(
  name='Dependency',
  full_name='dmir.Dependency',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='operation', full_name='dmir.Dependency.operation', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='argument', full_name='dmir.Dependency.argument', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result', full_name='dmir.Dependency.result', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='attribute', full_name='dmir.Dependency.attribute', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=662,
  serialized_end=763,
)


_GRAPH = _descriptor.Descriptor(
  name='Graph',
  full_name='dmir.Graph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='dmir.Graph.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='op_type', full_name='dmir.Graph.op_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input', full_name='dmir.Graph.input', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output', full_name='dmir.Graph.output', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='intermediate', full_name='dmir.Graph.intermediate', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='subgraph', full_name='dmir.Graph.subgraph', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dependency', full_name='dmir.Graph.dependency', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='device', full_name='dmir.Graph.device', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='dmir.Graph.metadata', index=8,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=766,
  serialized_end=1002,
)

_TENSOR.fields_by_name['format'].enum_type = _NUMERICALFORMAT
_ATTRIBUTE.fields_by_name['kind'].enum_type = _ATTRIBUTE_ATTRIBUTEKIND
_ATTRIBUTE_ATTRIBUTEKIND.containing_type = _ATTRIBUTE
_DEPENDENCY.fields_by_name['attribute'].message_type = _ATTRIBUTE
_GRAPH.fields_by_name['input'].message_type = _TENSOR
_GRAPH.fields_by_name['output'].message_type = _TENSOR
_GRAPH.fields_by_name['intermediate'].message_type = _TENSOR
_GRAPH.fields_by_name['subgraph'].message_type = _GRAPH
_GRAPH.fields_by_name['dependency'].message_type = _DEPENDENCY
DESCRIPTOR.message_types_by_name['Tensor'] = _TENSOR
DESCRIPTOR.message_types_by_name['Attribute'] = _ATTRIBUTE
DESCRIPTOR.message_types_by_name['Dependency'] = _DEPENDENCY
DESCRIPTOR.message_types_by_name['Graph'] = _GRAPH
DESCRIPTOR.enum_types_by_name['NumericalFormat'] = _NUMERICALFORMAT

Tensor = _reflection.GeneratedProtocolMessageType('Tensor', (_message.Message,), dict(
  DESCRIPTOR = _TENSOR,
  __module__ = 'dmir_pb2'
  # @@protoc_insertion_point(class_scope:dmir.Tensor)
  ))
_sym_db.RegisterMessage(Tensor)

Attribute = _reflection.GeneratedProtocolMessageType('Attribute', (_message.Message,), dict(
  DESCRIPTOR = _ATTRIBUTE,
  __module__ = 'dmir_pb2'
  # @@protoc_insertion_point(class_scope:dmir.Attribute)
  ))
_sym_db.RegisterMessage(Attribute)

Dependency = _reflection.GeneratedProtocolMessageType('Dependency', (_message.Message,), dict(
  DESCRIPTOR = _DEPENDENCY,
  __module__ = 'dmir_pb2'
  # @@protoc_insertion_point(class_scope:dmir.Dependency)
  ))
_sym_db.RegisterMessage(Dependency)

Graph = _reflection.GeneratedProtocolMessageType('Graph', (_message.Message,), dict(
  DESCRIPTOR = _GRAPH,
  __module__ = 'dmir_pb2'
  # @@protoc_insertion_point(class_scope:dmir.Graph)
  ))
_sym_db.RegisterMessage(Graph)


_NUMERICALFORMAT.has_options = True
_NUMERICALFORMAT._options = _descriptor._ParseOptions(descriptor_pb2.EnumOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
