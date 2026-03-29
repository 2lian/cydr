# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

from cpython.bytearray cimport PyByteArray_AS_STRING
from cpython.bytes cimport (
    PyBytes_AS_STRING,
    PyBytes_FromStringAndSize,
    PyBytes_GET_SIZE,
)
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.string cimport memcpy

cimport cython
cimport numpy as cnp
import numpy as np


cnp.import_array()


cdef int ENCAPSULATION_HEADER_SIZE = 4
cdef int CDR_ALIGN_MAX = 8

# This prototype assumes the host machine is little-endian.


cdef inline Py_ssize_t align_position(
    Py_ssize_t pos,
    int alignment,
    Py_ssize_t align_offset,
    int align_max,
) noexcept:
    cdef int effective_alignment = alignment
    if effective_alignment > align_max:
        effective_alignment = align_max
    return ((pos - align_offset + effective_alignment - 1) & ~(effective_alignment - 1)) + align_offset


cdef inline Py_ssize_t write_uint32_raw(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint32_t value,
) noexcept:
    memcpy(buffer + pos, &value, cython.sizeof(uint32_t))
    return pos + cython.sizeof(uint32_t)


cdef inline void write_encapsulation_header(unsigned char* buffer) noexcept:
    cdef uint32_t header = 0x00000100
    write_uint32_raw(buffer, 0, header)


cdef inline const unsigned char* data_ptr(const unsigned char[::1] data) noexcept:
    if data.shape[0] == 0:
        return cython.NULL
    return &data[0]


cdef inline void require_available(
    Py_ssize_t buffer_length,
    Py_ssize_t pos,
    Py_ssize_t byte_count,
) except *:
    if pos < 0 or byte_count < 0 or pos + byte_count > buffer_length:
        raise ValueError("Buffer too short while deserializing.")


cdef inline void require_consumed(
    const unsigned char[::1] data,
    Py_ssize_t pos,
) except *:
    if pos != data.shape[0]:
        raise ValueError("Trailing bytes after deserializing.")


cdef inline Py_ssize_t validate_encapsulation_header(
    const unsigned char[::1] data,
) except -1:
    require_available(data.shape[0], 0, ENCAPSULATION_HEADER_SIZE)
    if data[0] != 0 or data[1] != 1 or data[2] != 0 or data[3] != 0:
        raise ValueError("Unsupported encapsulation header.")
    return ENCAPSULATION_HEADER_SIZE


cdef inline uint32_t read_uint32_raw_value(
    const unsigned char[::1] data,
    Py_ssize_t pos,
) except? 0:
    cdef uint32_t value = 0
    require_available(data.shape[0], pos, cython.sizeof(uint32_t))
    memcpy(&value, data_ptr(data) + pos, cython.sizeof(uint32_t))
    return value


cdef inline tuple read_aligned_scalar_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
    object dtype,
):
    pos = align_position(pos, alignment, align_offset, CDR_ALIGN_MAX)
    require_available(data.shape[0], pos, itemsize)
    return np.frombuffer(data, dtype=dtype, count=1, offset=pos)[0], pos + itemsize


cdef inline tuple read_primitive_array_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
    object dtype,
):
    cdef Py_ssize_t byte_count = count * itemsize
    if count > 0:
        pos = align_position(pos, alignment, align_offset, CDR_ALIGN_MAX)
        require_available(data.shape[0], pos, byte_count)
        return np.frombuffer(data, dtype=dtype, count=count, offset=pos).copy(), pos + byte_count
    return np.empty(0, dtype=dtype), pos


cdef inline tuple read_primitive_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
    object dtype,
):
    cdef uint32_t count
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    count = read_uint32_raw_value(data, pos)
    pos += 4
    return read_primitive_array_object(
        data,
        pos,
        count,
        itemsize,
        alignment,
        align_offset,
        dtype,
    )


cdef inline tuple read_string_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    cdef uint32_t byte_count
    cdef Py_ssize_t string_size

    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    byte_count = read_uint32_raw_value(data, pos)
    pos += 4

    string_size = <Py_ssize_t> byte_count
    if string_size <= 0:
        raise ValueError("Invalid CDR string length.")

    require_available(data.shape[0], pos, string_size)
    if data[pos + string_size - 1] != 0:
        raise ValueError("CDR string is missing a trailing NUL byte.")

    return (
        PyBytes_FromStringAndSize(
            <const char*> (data_ptr(data) + pos),
            string_size - 1,
        ),
        pos + string_size,
    )


cdef inline tuple read_string_array_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t align_offset,
):
    cdef Py_ssize_t index
    cdef list values = []
    cdef object value

    for index in range(count):
        value, pos = read_string_object(data, pos, align_offset)
        values.append(value)

    return values, pos


cdef inline tuple read_string_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    cdef uint32_t count

    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    count = read_uint32_raw_value(data, pos)
    pos += 4

    return read_string_array_object(data, pos, count, align_offset)


cdef inline Py_ssize_t bytes_size(bytes value) noexcept:
    return PyBytes_GET_SIZE(value)


cdef inline Py_ssize_t write_raw_bytes(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t byte_count,
) noexcept:
    if byte_count > 0:
        memcpy(buffer + pos, data, <size_t> byte_count)
    return pos + byte_count


cdef inline Py_ssize_t write_aligned_value(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t byte_count,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, alignment, align_offset, CDR_ALIGN_MAX)
    return write_raw_bytes(buffer, pos, data, byte_count)


cdef inline Py_ssize_t write_bool(
    unsigned char* buffer,
    Py_ssize_t pos,
    bint value,
) noexcept:
    buffer[pos] = <unsigned char> value
    return pos + 1


cdef inline Py_ssize_t write_bytes_value(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
) noexcept:
    cdef const char* data = PyBytes_AS_STRING(value)
    return write_raw_bytes(buffer, pos, data, bytes_size(value))


cdef inline Py_ssize_t write_string(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
    Py_ssize_t align_offset,
) noexcept:
    cdef Py_ssize_t size = bytes_size(value)
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    pos = write_uint32_raw(buffer, pos, <uint32_t> (size + 1))
    pos = write_bytes_value(buffer, pos, value)
    buffer[pos] = 0
    return pos + 1


cdef inline Py_ssize_t advance_primitive_array_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    if count > 0:
        pos = align_position(pos, alignment, align_offset, CDR_ALIGN_MAX)
        pos += count * itemsize
    return pos


cdef inline Py_ssize_t advance_primitive_sequence_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    pos += 4
    return advance_primitive_array_position(pos, count, itemsize, alignment, align_offset)


cdef inline Py_ssize_t write_primitive_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    cdef Py_ssize_t byte_count
    if count > 0:
        pos = align_position(pos, alignment, align_offset, CDR_ALIGN_MAX)
        byte_count = count * itemsize
        pos = write_raw_bytes(buffer, pos, data, byte_count)
    return pos


cdef inline Py_ssize_t write_primitive_sequence(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    pos = write_uint32_raw(buffer, pos, <uint32_t> count)
    return write_primitive_array(buffer, pos, data, count, itemsize, alignment, align_offset)


cdef inline Py_ssize_t advance_string_array_position(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    cdef Py_ssize_t index
    cdef Py_ssize_t count = PyList_GET_SIZE(values)
    cdef bytes item
    for index in range(count):
        item = <bytes> PyList_GET_ITEM(values, index)
        pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
        pos += 4 + bytes_size(item) + 1
    return pos


cdef inline Py_ssize_t advance_string_sequence_position(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    pos += 4
    return advance_string_array_position(pos, values, align_offset)


cdef inline Py_ssize_t write_string_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    cdef Py_ssize_t index
    cdef Py_ssize_t count = PyList_GET_SIZE(values)
    cdef bytes item
    for index in range(count):
        item = <bytes> PyList_GET_ITEM(values, index)
        pos = write_string(buffer, pos, item, align_offset)
    return pos


cdef inline Py_ssize_t write_string_sequence(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    pos = write_uint32_raw(buffer, pos, <uint32_t> PyList_GET_SIZE(values))
    return write_string_array(buffer, pos, values, align_offset)


cdef inline Py_ssize_t write_boolean_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bint value,
) noexcept:
    return write_bool(buffer, pos, value)


cdef inline Py_ssize_t write_byte_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t write_int8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int8_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int8_t), 1, align_offset)


cdef inline Py_ssize_t write_uint8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t write_int16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int16_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int16_t), 2, align_offset)


cdef inline Py_ssize_t write_uint16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint16_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint16_t), 2, align_offset)


cdef inline Py_ssize_t write_int32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int32_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int32_t), 4, align_offset)


cdef inline Py_ssize_t write_uint32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint32_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint32_t), 4, align_offset)


cdef inline Py_ssize_t write_int64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int64_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int64_t), 8, align_offset)


cdef inline Py_ssize_t write_uint64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint64_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint64_t), 8, align_offset)


cdef inline Py_ssize_t write_float32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float32_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(cnp.float32_t), 4, align_offset)


cdef inline Py_ssize_t write_float64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float64_t value,
    Py_ssize_t align_offset,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(cnp.float64_t), 8, align_offset)


cdef inline Py_ssize_t write_string_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
    Py_ssize_t align_offset,
) noexcept:
    return write_string(buffer, pos, value, align_offset)


cdef inline tuple read_boolean_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
):
    require_available(data.shape[0], pos, 1)
    return bool(data[pos] != 0), pos + 1


cdef inline tuple read_byte_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(uint8_t),
        1,
        align_offset,
        np.uint8,
    )


cdef inline tuple read_int8_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(int8_t),
        1,
        align_offset,
        np.int8,
    )


cdef inline tuple read_uint8_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(uint8_t),
        1,
        align_offset,
        np.uint8,
    )


cdef inline tuple read_int16_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(int16_t),
        2,
        align_offset,
        np.int16,
    )


cdef inline tuple read_uint16_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(uint16_t),
        2,
        align_offset,
        np.uint16,
    )


cdef inline tuple read_int32_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(int32_t),
        4,
        align_offset,
        np.int32,
    )


cdef inline tuple read_uint32_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(uint32_t),
        4,
        align_offset,
        np.uint32,
    )


cdef inline tuple read_int64_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(int64_t),
        8,
        align_offset,
        np.int64,
    )


cdef inline tuple read_uint64_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(uint64_t),
        8,
        align_offset,
        np.uint64,
    )


cdef inline tuple read_float32_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(cnp.float32_t),
        4,
        align_offset,
        np.float32,
    )


cdef inline tuple read_float64_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_aligned_scalar_object(
        data,
        pos,
        cython.sizeof(cnp.float64_t),
        8,
        align_offset,
        np.float64,
    )


cdef inline tuple read_string_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_string_object(data, pos, align_offset)


cdef inline const void* bool_sequence_ptr(const cnp.npy_bool[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* uint8_view_ptr(const uint8_t[::1] values) noexcept:
    return &values[0]


cdef inline const void* int8_sequence_ptr(const int8_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* int16_sequence_ptr(const int16_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* uint16_view_ptr(const uint16_t[::1] values) noexcept:
    return &values[0]


cdef inline const void* int32_sequence_ptr(const int32_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* uint32_view_ptr(const uint32_t[::1] values) noexcept:
    return &values[0]


cdef inline const void* int64_sequence_ptr(const int64_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* uint64_view_ptr(const uint64_t[::1] values) noexcept:
    return &values[0]


cdef inline const void* float32_sequence_ptr(const cnp.float32_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* float64_sequence_ptr(const cnp.float64_t[::1] values) noexcept:
    if values.shape[0] == 0:
        return cython.NULL
    return &values[0]


cdef inline const void* float64_view_ptr(const cnp.float64_t[::1] values) noexcept:
    return &values[0]


cdef inline Py_ssize_t write_bool_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        bool_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.npy_bool),
        1,
        align_offset,
    )


cdef inline Py_ssize_t write_byte_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint8_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint8_t),
        1,
        align_offset,
    )


cdef inline Py_ssize_t write_int8_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int8_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int8_t),
        1,
        align_offset,
    )


cdef inline Py_ssize_t write_uint8_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint8_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint8_t),
        1,
        align_offset,
    )


cdef inline Py_ssize_t write_int16_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int16_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int16_t),
        2,
        align_offset,
    )


cdef inline Py_ssize_t write_uint16_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint16_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint16_t),
        2,
        align_offset,
    )


cdef inline Py_ssize_t write_int32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int32_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int32_t),
        4,
        align_offset,
    )


cdef inline Py_ssize_t write_uint32_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint32_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint32_t),
        4,
        align_offset,
    )


cdef inline Py_ssize_t write_int64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int64_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int64_t),
        8,
        align_offset,
    )


cdef inline Py_ssize_t write_uint64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint64_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint64_t),
        8,
        align_offset,
    )


cdef inline Py_ssize_t write_float32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        float32_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float32_t),
        4,
        align_offset,
    )


cdef inline Py_ssize_t write_float64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        float64_view_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float64_t),
        8,
        align_offset,
    )


cdef inline Py_ssize_t write_float64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        float64_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float64_t),
        8,
        align_offset,
    )


cdef inline Py_ssize_t write_text_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    return write_string_array(buffer, pos, values, align_offset)


cdef inline Py_ssize_t write_text_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    return write_string_sequence(buffer, pos, values, align_offset)


cdef inline tuple read_bool_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.npy_bool),
        1,
        align_offset,
        np.bool_,
    )


cdef inline tuple read_byte_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        3,
        cython.sizeof(uint8_t),
        1,
        align_offset,
        np.uint8,
    )


cdef inline tuple read_int8_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int8_t),
        1,
        align_offset,
        np.int8,
    )


cdef inline tuple read_uint8_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        3,
        cython.sizeof(uint8_t),
        1,
        align_offset,
        np.uint8,
    )


cdef inline tuple read_int16_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int16_t),
        2,
        align_offset,
        np.int16,
    )


cdef inline tuple read_uint16_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint16_t),
        2,
        align_offset,
        np.uint16,
    )


cdef inline tuple read_int32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int32_t),
        4,
        align_offset,
        np.int32,
    )


cdef inline tuple read_uint32_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint32_t),
        4,
        align_offset,
        np.uint32,
    )


cdef inline tuple read_int64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int64_t),
        8,
        align_offset,
        np.int64,
    )


cdef inline tuple read_uint64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint64_t),
        8,
        align_offset,
        np.uint64,
    )


cdef inline tuple read_float32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.float32_t),
        4,
        align_offset,
        np.float32,
    )


cdef inline tuple read_float64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(cnp.float64_t),
        8,
        align_offset,
        np.float64,
    )


cdef inline tuple read_float64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.float64_t),
        8,
        align_offset,
        np.float64,
    )


cdef inline tuple read_text_array_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_string_array_object(data, pos, 2, align_offset)


cdef inline tuple read_text_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t pos,
    Py_ssize_t align_offset,
):
    return read_string_sequence_object(data, pos, align_offset)


cdef inline void require_fixed_length(
    Py_ssize_t actual_count,
    Py_ssize_t expected_count,
    str field_name,
) except *:
    if actual_count != expected_count:
        raise ValueError(f"{field_name} must have length {expected_count}")


cdef inline Py_ssize_t advance_scalar_field(
    Py_ssize_t pos,
    Py_ssize_t itemsize,
    int alignment,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_array_position(pos, 1, itemsize, alignment, align_offset)


cdef inline Py_ssize_t advance_string_field(
    Py_ssize_t pos,
    bytes value,
    Py_ssize_t align_offset,
) noexcept:
    pos = align_position(pos, 4, align_offset, CDR_ALIGN_MAX)
    return pos + 4 + bytes_size(value) + 1


cdef inline Py_ssize_t advance_boolean_field(Py_ssize_t pos) noexcept:
    return pos + 1


cdef inline Py_ssize_t advance_byte_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t advance_int8_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int8_t), 1, align_offset)


cdef inline Py_ssize_t advance_uint8_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t advance_int16_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int16_t), 2, align_offset)


cdef inline Py_ssize_t advance_uint16_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint16_t), 2, align_offset)


cdef inline Py_ssize_t advance_int32_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int32_t), 4, align_offset)


cdef inline Py_ssize_t advance_uint32_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint32_t), 4, align_offset)


cdef inline Py_ssize_t advance_int64_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int64_t), 8, align_offset)


cdef inline Py_ssize_t advance_uint64_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint64_t), 8, align_offset)


cdef inline Py_ssize_t advance_float32_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(cnp.float32_t), 4, align_offset)


cdef inline Py_ssize_t advance_float64_field(
    Py_ssize_t pos,
    Py_ssize_t align_offset,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(cnp.float64_t), 8, align_offset)


cdef inline Py_ssize_t advance_bool_sequence_field(
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.npy_bool), 1, align_offset
    )


cdef inline Py_ssize_t advance_byte_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 3, "byte_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t advance_int8_sequence_field(
    Py_ssize_t pos,
    const int8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int8_t), 1, align_offset)


cdef inline Py_ssize_t advance_uint8_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 3, "uint8_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint8_t), 1, align_offset)


cdef inline Py_ssize_t advance_int16_sequence_field(
    Py_ssize_t pos,
    const int16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int16_t), 2, align_offset)


cdef inline Py_ssize_t advance_uint16_array_field(
    Py_ssize_t pos,
    const uint16_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint16_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint16_t), 2, align_offset)


cdef inline Py_ssize_t advance_int32_sequence_field(
    Py_ssize_t pos,
    const int32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int32_t), 4, align_offset)


cdef inline Py_ssize_t advance_uint32_array_field(
    Py_ssize_t pos,
    const uint32_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint32_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint32_t), 4, align_offset)


cdef inline Py_ssize_t advance_int64_sequence_field(
    Py_ssize_t pos,
    const int64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int64_t), 8, align_offset)


cdef inline Py_ssize_t advance_uint64_array_field(
    Py_ssize_t pos,
    const uint64_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint64_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint64_t), 8, align_offset)


cdef inline Py_ssize_t advance_float32_sequence_field(
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.float32_t), 4, align_offset
    )


cdef inline Py_ssize_t advance_float64_array_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(values.shape[0], 2, "float64_array")
    return advance_primitive_array_position(
        pos, values.shape[0], cython.sizeof(cnp.float64_t), 8, align_offset
    )


cdef inline Py_ssize_t advance_float64_sequence_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.float64_t), 8, align_offset
    )


cdef inline Py_ssize_t advance_text_array_field(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) except -1:
    require_fixed_length(PyList_GET_SIZE(values), 2, "text_array")
    return advance_string_array_position(pos, values, align_offset)


cdef inline Py_ssize_t advance_text_sequence_field(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept:
    return advance_string_sequence_position(pos, values, align_offset)


cpdef Py_ssize_t compute_serialized_size_every_supported_schema(
    bint boolean_value,
    uint8_t byte_value,
    int8_t signed_int8,
    uint8_t unsigned_int8,
    int16_t signed_int16,
    uint16_t unsigned_int16,
    int32_t signed_int32,
    uint32_t unsigned_int32,
    int64_t signed_int64,
    uint64_t unsigned_int64,
    cnp.float32_t float32_value,
    cnp.float64_t float64_value,
    bytes text,
    int32_t header_stamp_sec,
    uint32_t header_stamp_nanosec,
    bytes header_frame_id,
    const cnp.npy_bool[::1] bool_sequence,
    const uint8_t[::1] byte_array,
    const int8_t[::1] int8_sequence,
    const uint8_t[::1] uint8_array,
    const int16_t[::1] int16_sequence,
    const uint16_t[::1] uint16_array,
    const int32_t[::1] int32_sequence,
    const uint32_t[::1] uint32_array,
    const int64_t[::1] int64_sequence,
    const uint64_t[::1] uint64_array,
    const cnp.float32_t[::1] float32_sequence,
    const cnp.float64_t[::1] float64_array,
    list text_array,
    list text_sequence,
) except -1:
    cdef Py_ssize_t pos = ENCAPSULATION_HEADER_SIZE
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE

    pos = advance_boolean_field(pos)
    pos = advance_byte_field(pos, align_offset)
    pos = advance_int8_field(pos, align_offset)
    pos = advance_uint8_field(pos, align_offset)
    pos = advance_int16_field(pos, align_offset)
    pos = advance_uint16_field(pos, align_offset)
    pos = advance_int32_field(pos, align_offset)
    pos = advance_uint32_field(pos, align_offset)
    pos = advance_int64_field(pos, align_offset)
    pos = advance_uint64_field(pos, align_offset)
    pos = advance_float32_field(pos, align_offset)
    pos = advance_float64_field(pos, align_offset)
    pos = advance_string_field(pos, text, align_offset)
    pos = advance_int32_field(pos, align_offset)
    pos = advance_uint32_field(pos, align_offset)
    pos = advance_string_field(pos, header_frame_id, align_offset)

    pos = advance_bool_sequence_field(pos, bool_sequence, align_offset)
    pos = advance_byte_array_field(pos, byte_array, align_offset)
    pos = advance_int8_sequence_field(pos, int8_sequence, align_offset)
    pos = advance_uint8_array_field(pos, uint8_array, align_offset)
    pos = advance_int16_sequence_field(pos, int16_sequence, align_offset)
    pos = advance_uint16_array_field(pos, uint16_array, align_offset)
    pos = advance_int32_sequence_field(pos, int32_sequence, align_offset)
    pos = advance_uint32_array_field(pos, uint32_array, align_offset)
    pos = advance_int64_sequence_field(pos, int64_sequence, align_offset)
    pos = advance_uint64_array_field(pos, uint64_array, align_offset)
    pos = advance_float32_sequence_field(pos, float32_sequence, align_offset)
    pos = advance_float64_array_field(pos, float64_array, align_offset)
    pos = advance_text_array_field(pos, text_array, align_offset)
    pos = advance_text_sequence_field(pos, text_sequence, align_offset)
    return pos


cpdef bytearray serialize_every_supported_schema(
    bint boolean_value,
    uint8_t byte_value,
    int8_t signed_int8,
    uint8_t unsigned_int8,
    int16_t signed_int16,
    uint16_t unsigned_int16,
    int32_t signed_int32,
    uint32_t unsigned_int32,
    int64_t signed_int64,
    uint64_t unsigned_int64,
    cnp.float32_t float32_value,
    cnp.float64_t float64_value,
    bytes text,
    int32_t header_stamp_sec,
    uint32_t header_stamp_nanosec,
    bytes header_frame_id,
    const cnp.npy_bool[::1] bool_sequence,
    const uint8_t[::1] byte_array,
    const int8_t[::1] int8_sequence,
    const uint8_t[::1] uint8_array,
    const int16_t[::1] int16_sequence,
    const uint16_t[::1] uint16_array,
    const int32_t[::1] int32_sequence,
    const uint32_t[::1] uint32_array,
    const int64_t[::1] int64_sequence,
    const uint64_t[::1] uint64_array,
    const cnp.float32_t[::1] float32_sequence,
    const cnp.float64_t[::1] float64_array,
    list text_array,
    list text_sequence,
):
    cdef Py_ssize_t total_size = compute_serialized_size_every_supported_schema(
        boolean_value,
        byte_value,
        signed_int8,
        unsigned_int8,
        signed_int16,
        unsigned_int16,
        signed_int32,
        unsigned_int32,
        signed_int64,
        unsigned_int64,
        float32_value,
        float64_value,
        text,
        header_stamp_sec,
        header_stamp_nanosec,
        header_frame_id,
        bool_sequence,
        byte_array,
        int8_sequence,
        uint8_array,
        int16_sequence,
        uint16_array,
        int32_sequence,
        uint32_array,
        int64_sequence,
        uint64_array,
        float32_sequence,
        float64_array,
        text_array,
        text_sequence,
    )
    cdef bytearray output = bytearray(total_size)
    cdef unsigned char* buffer = <unsigned char*> PyByteArray_AS_STRING(output)
    cdef Py_ssize_t pos = ENCAPSULATION_HEADER_SIZE
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE

    write_encapsulation_header(buffer)

    pos = write_boolean_field(buffer, pos, boolean_value)
    pos = write_byte_field(buffer, pos, byte_value, align_offset)
    pos = write_int8_field(buffer, pos, signed_int8, align_offset)
    pos = write_uint8_field(buffer, pos, unsigned_int8, align_offset)
    pos = write_int16_field(buffer, pos, signed_int16, align_offset)
    pos = write_uint16_field(buffer, pos, unsigned_int16, align_offset)
    pos = write_int32_field(buffer, pos, signed_int32, align_offset)
    pos = write_uint32_field(buffer, pos, unsigned_int32, align_offset)
    pos = write_int64_field(buffer, pos, signed_int64, align_offset)
    pos = write_uint64_field(buffer, pos, unsigned_int64, align_offset)
    pos = write_float32_field(buffer, pos, float32_value, align_offset)
    pos = write_float64_field(buffer, pos, float64_value, align_offset)
    pos = write_string_field(buffer, pos, text, align_offset)
    pos = write_int32_field(buffer, pos, header_stamp_sec, align_offset)
    pos = write_uint32_field(buffer, pos, header_stamp_nanosec, align_offset)
    pos = write_string_field(buffer, pos, header_frame_id, align_offset)

    pos = write_bool_sequence_field(buffer, pos, bool_sequence, align_offset)
    pos = write_byte_array_field(buffer, pos, byte_array, align_offset)
    pos = write_int8_sequence_field(buffer, pos, int8_sequence, align_offset)
    pos = write_uint8_array_field(buffer, pos, uint8_array, align_offset)
    pos = write_int16_sequence_field(buffer, pos, int16_sequence, align_offset)
    pos = write_uint16_array_field(buffer, pos, uint16_array, align_offset)
    pos = write_int32_sequence_field(buffer, pos, int32_sequence, align_offset)
    pos = write_uint32_array_field(buffer, pos, uint32_array, align_offset)
    pos = write_int64_sequence_field(buffer, pos, int64_sequence, align_offset)
    pos = write_uint64_array_field(buffer, pos, uint64_array, align_offset)
    pos = write_float32_sequence_field(buffer, pos, float32_sequence, align_offset)
    pos = write_float64_array_field(buffer, pos, float64_array, align_offset)
    pos = write_text_array_field(buffer, pos, text_array, align_offset)
    pos = write_text_sequence_field(buffer, pos, text_sequence, align_offset)
    return output


cpdef dict deserialize_every_supported_schema(const unsigned char[::1] data):
    cdef Py_ssize_t pos = validate_encapsulation_header(data)
    cdef Py_ssize_t align_offset = ENCAPSULATION_HEADER_SIZE
    cdef object boolean_value
    cdef object byte_value
    cdef object signed_int8
    cdef object unsigned_int8
    cdef object signed_int16
    cdef object unsigned_int16
    cdef object signed_int32
    cdef object unsigned_int32
    cdef object signed_int64
    cdef object unsigned_int64
    cdef object float32_value
    cdef object float64_value
    cdef object text
    cdef object header_stamp_sec
    cdef object header_stamp_nanosec
    cdef object header_frame_id
    cdef object bool_sequence
    cdef object byte_array
    cdef object int8_sequence
    cdef object uint8_array
    cdef object int16_sequence
    cdef object uint16_array
    cdef object int32_sequence
    cdef object uint32_array
    cdef object int64_sequence
    cdef object uint64_array
    cdef object float32_sequence
    cdef object float64_array
    cdef object text_array
    cdef object text_sequence

    boolean_value, pos = read_boolean_field(data, pos)
    byte_value, pos = read_byte_field(data, pos, align_offset)
    signed_int8, pos = read_int8_field(data, pos, align_offset)
    unsigned_int8, pos = read_uint8_field(data, pos, align_offset)
    signed_int16, pos = read_int16_field(data, pos, align_offset)
    unsigned_int16, pos = read_uint16_field(data, pos, align_offset)
    signed_int32, pos = read_int32_field(data, pos, align_offset)
    unsigned_int32, pos = read_uint32_field(data, pos, align_offset)
    signed_int64, pos = read_int64_field(data, pos, align_offset)
    unsigned_int64, pos = read_uint64_field(data, pos, align_offset)
    float32_value, pos = read_float32_field(data, pos, align_offset)
    float64_value, pos = read_float64_field(data, pos, align_offset)
    text, pos = read_string_field(data, pos, align_offset)
    header_stamp_sec, pos = read_int32_field(data, pos, align_offset)
    header_stamp_nanosec, pos = read_uint32_field(data, pos, align_offset)
    header_frame_id, pos = read_string_field(data, pos, align_offset)
    bool_sequence, pos = read_bool_sequence_field(data, pos, align_offset)
    byte_array, pos = read_byte_array_field(data, pos, align_offset)
    int8_sequence, pos = read_int8_sequence_field(data, pos, align_offset)
    uint8_array, pos = read_uint8_array_field(data, pos, align_offset)
    int16_sequence, pos = read_int16_sequence_field(data, pos, align_offset)
    uint16_array, pos = read_uint16_array_field(data, pos, align_offset)
    int32_sequence, pos = read_int32_sequence_field(data, pos, align_offset)
    uint32_array, pos = read_uint32_array_field(data, pos, align_offset)
    int64_sequence, pos = read_int64_sequence_field(data, pos, align_offset)
    uint64_array, pos = read_uint64_array_field(data, pos, align_offset)
    float32_sequence, pos = read_float32_sequence_field(data, pos, align_offset)
    float64_array, pos = read_float64_array_field(data, pos, align_offset)
    text_array, pos = read_text_array_field(data, pos, align_offset)
    text_sequence, pos = read_text_sequence_field(data, pos, align_offset)
    require_consumed(data, pos)

    return {
        "boolean_value": boolean_value,
        "byte_value": byte_value,
        "signed_int8": signed_int8,
        "unsigned_int8": unsigned_int8,
        "signed_int16": signed_int16,
        "unsigned_int16": unsigned_int16,
        "signed_int32": signed_int32,
        "unsigned_int32": unsigned_int32,
        "signed_int64": signed_int64,
        "unsigned_int64": unsigned_int64,
        "float32_value": float32_value,
        "float64_value": float64_value,
        "text": text,
        "header": {
            "stamp": {
                "sec": header_stamp_sec,
                "nanosec": header_stamp_nanosec,
            },
            "frame_id": header_frame_id,
        },
        "bool_sequence": bool_sequence,
        "byte_array": byte_array,
        "int8_sequence": int8_sequence,
        "uint8_array": uint8_array,
        "int16_sequence": int16_sequence,
        "uint16_array": uint16_array,
        "int32_sequence": int32_sequence,
        "uint32_array": uint32_array,
        "int64_sequence": int64_sequence,
        "uint64_array": uint64_array,
        "float32_sequence": float32_sequence,
        "float64_array": float64_array,
        "text_array": text_array,
        "text_sequence": text_sequence,
    }
