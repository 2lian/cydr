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

cimport numpy as cnp


cdef int ENCAPSULATION_HEADER_SIZE

cdef void write_encapsulation_header(unsigned char* buffer) noexcept

cdef Py_ssize_t advance_boolean_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_byte_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_int8_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_uint8_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_int16_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_uint16_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_int32_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_uint32_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_int64_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_uint64_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_float32_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_float64_field(Py_ssize_t pos, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_string_field(Py_ssize_t pos, bytes value, Py_ssize_t align_offset) noexcept
cdef Py_ssize_t advance_bool_sequence_field(
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_byte_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_int8_sequence_field(
    Py_ssize_t pos,
    const int8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_uint8_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_int16_sequence_field(
    Py_ssize_t pos,
    const int16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_uint16_array_field(
    Py_ssize_t pos,
    const uint16_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_int32_sequence_field(
    Py_ssize_t pos,
    const int32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_uint32_array_field(
    Py_ssize_t pos,
    const uint32_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_int64_sequence_field(
    Py_ssize_t pos,
    const int64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_uint64_array_field(
    Py_ssize_t pos,
    const uint64_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_float32_sequence_field(
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_float64_array_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_float64_sequence_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t advance_text_array_field(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) except -1
cdef Py_ssize_t advance_text_sequence_field(
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept

cdef Py_ssize_t write_boolean_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bint value,
) noexcept
cdef Py_ssize_t write_byte_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int8_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int16_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint16_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int32_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint32_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int64_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint64_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_float32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float32_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_float64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float64_t value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_string_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_bool_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_byte_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int8_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint8_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int16_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint16_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint16_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint32_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_int64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_uint64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_float32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_float64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_float64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_text_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept
cdef Py_ssize_t write_text_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    list values,
    Py_ssize_t align_offset,
) noexcept
