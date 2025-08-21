# PNNX Format Formal Specification

This document provides a comprehensive formal specification of the PNNX (PyTorch Neural Network eXchange) format, including both the `.pnnx.param` text format and the `.pnnx.bin` binary format.

## Table of Contents

1. [Overview](#overview)
2. [PNNX.PARAM Format](#pnnxparam-format)
3. [PNNX.BIN Format](#pnnxbin-format)
4. [Data Types](#data-types)
5. [Parameter Encoding](#parameter-encoding)
6. [Validation Rules](#validation-rules)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Overview

PNNX format consists of two files:
- **`.pnnx.param`**: Human-readable text file containing the computation graph structure
- **`.pnnx.bin`**: Binary file containing model weights and large tensor data

## PNNX.PARAM Format

### Grammar Specification (EBNF)

```ebnf
pnnx_file = magic_line header_line operator_line*

magic_line = "7767517" newline

header_line = operator_count whitespace operand_count newline

operator_line = operator_type whitespace 
                operator_name whitespace
                input_count whitespace
                output_count
                (whitespace operand_name)*
                (whitespace parameter)*
                newline

operator_type = identifier
operator_name = identifier  
input_count = integer
output_count = integer
operand_name = identifier
parameter = key "=" value

key = normal_key | attribute_key | shape_key | input_key
normal_key = identifier
attribute_key = "@" identifier
shape_key = "#" identifier  
input_key = "$" identifier

value = parameter_value | attribute_value | shape_value | input_value

identifier = letter (letter | digit | "_" | ".")*
integer = ["-"] digit+
float = ["-"] digit+ "." digit+ [("e"|"E") ["-"|"+"] digit+]
letter = "a".."z" | "A".."Z"
digit = "0".."9"
whitespace = " " | "\t"
newline = "\n" | "\r\n"
```

### File Structure

#### 1. Magic Number Line
```
7767517
```
- **Purpose**: File format identification
- **Type**: Fixed 32-bit integer
- **Value**: Must be exactly `7767517`
- **Encoding**: ASCII decimal representation

#### 2. Header Line
```
<operator_count> <operand_count>
```
- **operator_count**: Total number of operators in the graph
- **operand_count**: Total number of unique operands (tensors/blobs)
- **Separation**: Single space character
- **Range**: Non-negative integers (0 ≤ count ≤ 2³¹-1)

#### 3. Operator Lines
Each operator line follows this format:
```
<type> <name> <input_count> <output_count> [input_operands] [output_operands] [parameters]
```

**Field Specifications:**
- **type**: Operator type identifier (e.g., `nn.Conv2d`, `pnnx.Input`, `torch.add`)
- **name**: Unique operator instance name
- **input_count**: Number of input operands (0 ≤ input_count ≤ 255)
- **output_count**: Number of output operands (0 ≤ output_count ≤ 255)
- **input_operands**: Space-separated list of input operand names
- **output_operands**: Space-separated list of output operand names
- **parameters**: Space-separated key=value pairs

### Parameter Encoding

Parameters use different prefixes to indicate their type and purpose:

#### Regular Parameters (no prefix)
Basic operator parameters:
```
key=value
```
Examples:
- `bias=True`
- `stride=(2,2)`
- `in_channels=64`

#### Attribute Parameters (@ prefix)
Weight and tensor data references:
```
@key=shape_spec[type_spec]
```
Format: `@weight=(out,in,h,w)f32`

**Shape Specification:**
- Parentheses-enclosed comma-separated dimensions
- Special symbols:
  - `?`: Dynamic dimension (size unknown at compile time)
  - `%symbol`: Symbolic dimension with named reference

**Type Specification:**
- `f32`: 32-bit float
- `f64`: 64-bit double  
- `f16`: 16-bit half
- `i32`: 32-bit integer
- `i64`: 64-bit long
- `i16`: 16-bit short
- `i8`: 8-bit signed char
- `u8`: 8-bit unsigned char
- `bool`: Boolean
- `c64`: 64-bit complex (2×32-bit float)
- `c128`: 128-bit complex (2×64-bit double)
- `c32`: 32-bit complex (2×16-bit half)
- `bf16`: 16-bit bfloat

#### Shape Parameters (# prefix)
Operand shape information:
```
#key=shape_spec
```
Used to specify output tensor shapes when they cannot be inferred.

#### Input Parameters ($ prefix)
Input parameter mappings:
```
$key=value
```
Used to reference input operand parameters.

### Data Type System

#### Parameter Types
Parameters are encoded with the following type system:

| Type ID | Name | Description | Example Values |
|---------|------|-------------|----------------|
| 0 | null | No value | `None`, `()`, `[]` |
| 1 | bool | Boolean | `True`, `False` |
| 2 | int | 32-bit integer | `42`, `-123` |
| 3 | float | 32-bit float | `3.14`, `-2.5e-3` |
| 4 | string | String literal | `"relu"`, `same` |
| 5 | int_array | Integer array | `(1,2,3)`, `[224,224]` |
| 6 | float_array | Float array | `(0.1,0.2)`, `[1.0,2.0]` |
| 7 | string_array | String array | `("a","b")`, `["x","y"]` |
| 10 | complex | Complex number | `1+2j` |
| 11 | complex_array | Complex array | `(1+2j,3+4j)` |

#### Operand/Attribute Types

| Type ID | Name | Description | Element Size |
|---------|------|-------------|--------------|
| 0 | null | No data | 0 |
| 1 | f32 | 32-bit float | 4 bytes |
| 2 | f64 | 64-bit double | 8 bytes |
| 3 | f16 | 16-bit half | 2 bytes |
| 4 | i32 | 32-bit integer | 4 bytes |
| 5 | i64 | 64-bit long | 8 bytes |
| 6 | i16 | 16-bit short | 2 bytes |
| 7 | i8 | 8-bit signed char | 1 byte |
| 8 | u8 | 8-bit unsigned char | 1 byte |
| 9 | bool | Boolean | 1 byte |
| 10 | c64 | 64-bit complex | 8 bytes |
| 11 | c128 | 128-bit complex | 16 bytes |
| 12 | c32 | 32-bit complex | 4 bytes |
| 13 | bf16 | 16-bit bfloat | 2 bytes |

## PNNX.BIN Format

### Structure
The `.pnnx.bin` file is a ZIP archive using store-only mode (no compression) containing binary weight data.

### Archive Organization
```
pnnx.bin (ZIP archive)
├── operator_name1.weight
├── operator_name1.bias
├── operator_name2.weight
└── ...
```

### File Naming Convention
Weight files are named using the pattern:
```
{operator_name}.{attribute_name}
```

Examples:
- `conv_0.weight` - Weight tensor for operator named "conv_0"
- `conv_0.bias` - Bias tensor for operator named "conv_0"
- `bn_1.running_mean` - Running mean for BatchNorm operator "bn_1"

### Binary Data Format
Each weight file contains raw binary tensor data:
- **Byte Order**: Little-endian
- **Layout**: Row-major (C-style) memory layout
- **Alignment**: No padding between elements
- **Size**: `product(shape) * element_size` bytes

## Validation Rules

### File Level Validation
1. Magic number must be exactly `7767517`
2. Operator count must match actual number of operator lines
3. All referenced operand names must be defined
4. No circular dependencies in the computation graph

### Operator Level Validation
1. Input count must match number of input operand names
2. Output count must match number of output operand names
3. All output operand names must be unique across the entire graph
4. Input operand names must reference existing operands

### Parameter Level Validation
1. Attribute references (`@` prefix) must have corresponding binary data
2. Shape specifications must use valid dimension syntax
3. Type specifications must use valid type identifiers
4. Parameter values must conform to declared types

### Cross-File Validation
1. All `@` attributes in `.param` must have corresponding files in `.bin`
2. Binary file sizes must match declared tensor shapes and types
3. Binary data must be readable as specified type

## Special PNNX Operators

PNNX defines several built-in operators with special semantics:

### Core Operators
- **`pnnx.Input`**: Graph input nodes
  - Always has 0 inputs, 1 output
  - Defines input tensor specifications
  
- **`pnnx.Output`**: Graph output nodes
  - Always has 1 input, 0 outputs
  - Marks graph outputs

### Meta Operators  
- **`pnnx.Expression`**: Mathematical expressions
  - Uses `expr` parameter with special syntax
  - References inputs as `@0`, `@1`, etc.
  - Example: `expr=add(@0,mul(@1,2.0))`

- **`pnnx.Attribute`**: External attribute references
  - Links to external tensor data
  - Used for large constant tensors

- **`pnnx.SliceIndexes`**: Advanced indexing operations
  - Handles complex tensor slicing
  - Used for dynamic indexing patterns

### Expression Syntax
The `pnnx.Expression` operator uses a special expression language:

```
expr=operation(arg1,arg2,...)
```

**Supported operations:**
- Arithmetic: `add`, `sub`, `mul`, `div`, `mod`
- Math functions: `sin`, `cos`, `exp`, `log`, `sqrt`
- Tensor operations: `view`, `transpose`, `permute`
- Logic: `eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Argument types:**
- `@N`: Reference to N-th input operand
- `literal`: Numeric or string literal
- `[a,b,c]`: Array literal

## Error Handling

### Parse Errors
- Invalid magic number → "Invalid PNNX format"
- Malformed header → "Invalid operator/operand count"
- Invalid operator line → "Malformed operator definition"
- Missing required fields → "Missing operator field: {field}"

### Reference Errors
- Undefined operand reference → "Undefined operand: {name}"
- Missing binary data → "Missing weight data: {operator}.{attribute}"
- Type mismatch → "Type mismatch for {operator}.{attribute}"

### Validation Errors
- Circular dependency → "Circular dependency detected"
- Duplicate operand names → "Duplicate operand name: {name}"
- Invalid parameter syntax → "Invalid parameter: {key}={value}"

## Examples

### Minimal Example
```
7767517
2 1
pnnx.Input      input_0     0 1 x
pnnx.Output     output_0    1 0 x
```

### Convolution Example
```
7767517
3 3
pnnx.Input      input_0     0 1 x
nn.Conv2d       conv_0      1 1 x y bias=True in_channels=3 kernel_size=(3,3) out_channels=64 padding=(1,1) stride=(1,1) @bias=(64)f32 @weight=(64,3,3,3)f32
pnnx.Output     output_0    1 0 y
```

### Complex Parameter Example
```
nn.LSTM         lstm_0      1 2 x y z batch_first=True bidirectional=False dropout=0.0 hidden_size=128 input_size=256 num_layers=2 proj_size=0 @weight_hh_l0=(512,128)f32 @weight_ih_l0=(512,256)f32 @bias_hh_l0=(512)f32 @bias_ih_l0=(512)f32
```

### Shape with Symbolic Dimensions
```
torch.view      view_0      1 1 x y size=(-1,%batch_size,256) #size=(-1,%batch_size,256)
```

### Expression Operator Example
```
pnnx.Expression expr_0     2 1 a b c expr=add(@0,mul(@1,2.0))
```

### Parameter Type Examples
```
# Boolean parameters
dropout=True
training=False

# Numeric parameters  
eps=1e-05
momentum=0.1
num_features=256

# String parameters
mode=bilinear
padding_mode=zeros

# Array parameters
kernel_size=(3,3)
stride=[2,2]
dilation=(1,1,1)

# Complex array parameters
some_param=(1.0,2.0,3.0)
indices=[0,1,2,3]
```

## Version Compatibility

### Current Version
- **Magic Number**: `7767517`
- **Format Version**: 1.0
- **Compatibility**: This specification describes the current stable format

### Future Versions
Future versions of PNNX format may:
- Use different magic numbers for backward compatibility detection
- Extend parameter encoding while maintaining backward compatibility
- Add optional metadata sections

### Implementation Notes
- Parsers should validate the magic number first
- Unknown parameter prefixes should be treated as errors
- Unsupported data types should be rejected with clear error messages

This formal specification provides the complete definition of the PNNX format for implementation, validation, and interoperability purposes.