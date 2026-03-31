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
cdef int STRING_COLLECTION_MODE_NUMPY
cdef int STRING_COLLECTION_MODE_LIST
cdef int STRING_COLLECTION_MODE_STRING_DTYPE
cdef int STRING_COLLECTION_MODE_RAW

cdef void validate_encapsulation_header(const unsigned char[::1] data) except *
cdef void require_consumed(const unsigned char[::1] data, Py_ssize_t pos) except *
cdef void require_fixed_length(
    Py_ssize_t actual_count,
    Py_ssize_t expected_count,
    str field_name,
) except *

cdef void write_encapsulation_header(unsigned char* buffer) noexcept

cdef Py_ssize_t advance_primitive_array_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept
cdef Py_ssize_t advance_primitive_sequence_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept
cdef Py_ssize_t advance_string_array_position(
    Py_ssize_t pos,
    object values,
) except -1
cdef Py_ssize_t advance_string_sequence_position(
    Py_ssize_t pos,
    object values,
) except -1
cdef Py_ssize_t write_primitive_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept
cdef Py_ssize_t write_primitive_sequence(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept
cdef Py_ssize_t write_string_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1
cdef Py_ssize_t write_string_sequence(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1
cdef cnp.ndarray read_primitive_array_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    int type_num,
)
cdef cnp.ndarray read_primitive_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t itemsize,
    int alignment,
    int type_num,
)

ctypedef struct DecodedStringSpans:
    Py_ssize_t count
    const unsigned char* base
    Py_ssize_t* offsets
    Py_ssize_t* sizes

cdef class DecodedStringCollection:
    cdef DecodedStringSpans spans
    cdef object owner
    cdef bytes value_at(self, Py_ssize_t index)
    cdef void init(
        self,
        const unsigned char* base,
        object owner,
        Py_ssize_t count,
    ) except *
    cdef void set_item(
        self,
        Py_ssize_t index,
        Py_ssize_t offset,
        Py_ssize_t size,
    ) noexcept
    cpdef list to_list(self)
    cpdef cnp.ndarray to_numpy(self)
    cpdef cnp.ndarray to_numpy_string_dtype(self)
    cpdef object to_final(self, int string_collection_mode)

cdef DecodedStringCollection read_string_array_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t count,
)
cdef DecodedStringCollection read_string_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef Py_ssize_t string_collection_length(object values) except -1

cdef const void* bool_sequence_ptr(const cnp.npy_bool[::1] values) noexcept
cdef const void* uint8_view_ptr(const uint8_t[::1] values) noexcept
cdef const void* int8_sequence_ptr(const int8_t[::1] values) noexcept
cdef const void* int16_sequence_ptr(const int16_t[::1] values) noexcept
cdef const void* uint16_view_ptr(const uint16_t[::1] values) noexcept
cdef const void* int32_sequence_ptr(const int32_t[::1] values) noexcept
cdef const void* uint32_view_ptr(const uint32_t[::1] values) noexcept
cdef const void* int64_sequence_ptr(const int64_t[::1] values) noexcept
cdef const void* uint64_view_ptr(const uint64_t[::1] values) noexcept
cdef const void* float32_sequence_ptr(const cnp.float32_t[::1] values) noexcept
cdef const void* float64_sequence_ptr(const cnp.float64_t[::1] values) noexcept
cdef const void* float64_view_ptr(const cnp.float64_t[::1] values) noexcept

cdef Py_ssize_t advance_boolean_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_byte_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_int8_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_uint8_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_int16_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_uint16_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_int32_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_uint32_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_int64_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_uint64_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_float32_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_float64_field(Py_ssize_t pos) noexcept
cdef Py_ssize_t advance_string_field(Py_ssize_t pos, bytes value) noexcept
cdef Py_ssize_t advance_bool_sequence_field(
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
) noexcept
cdef Py_ssize_t advance_byte_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
) except -1
cdef Py_ssize_t advance_int8_sequence_field(
    Py_ssize_t pos,
    const int8_t[::1] values,
) noexcept
cdef Py_ssize_t advance_uint8_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
) except -1
cdef Py_ssize_t advance_int16_sequence_field(
    Py_ssize_t pos,
    const int16_t[::1] values,
) noexcept
cdef Py_ssize_t advance_uint16_array_field(
    Py_ssize_t pos,
    const uint16_t[::1] values,
) except -1
cdef Py_ssize_t advance_int32_sequence_field(
    Py_ssize_t pos,
    const int32_t[::1] values,
) noexcept
cdef Py_ssize_t advance_uint32_array_field(
    Py_ssize_t pos,
    const uint32_t[::1] values,
) except -1
cdef Py_ssize_t advance_int64_sequence_field(
    Py_ssize_t pos,
    const int64_t[::1] values,
) noexcept
cdef Py_ssize_t advance_uint64_array_field(
    Py_ssize_t pos,
    const uint64_t[::1] values,
) except -1
cdef Py_ssize_t advance_float32_sequence_field(
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
) noexcept
cdef Py_ssize_t advance_float64_array_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) except -1
cdef Py_ssize_t advance_float64_sequence_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept
cdef Py_ssize_t advance_text_array_field(
    Py_ssize_t pos,
    object values,
) except -1
cdef Py_ssize_t advance_text_sequence_field(
    Py_ssize_t pos,
    object values,
) except -1

cdef Py_ssize_t write_boolean_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bint value,
) noexcept
cdef Py_ssize_t write_byte_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
) noexcept
cdef Py_ssize_t write_int8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int8_t value,
) noexcept
cdef Py_ssize_t write_uint8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
) noexcept
cdef Py_ssize_t write_int16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int16_t value,
) noexcept
cdef Py_ssize_t write_uint16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint16_t value,
) noexcept
cdef Py_ssize_t write_int32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int32_t value,
) noexcept
cdef Py_ssize_t write_uint32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint32_t value,
) noexcept
cdef Py_ssize_t write_int64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int64_t value,
) noexcept
cdef Py_ssize_t write_uint64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint64_t value,
) noexcept
cdef Py_ssize_t write_float32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float32_t value,
) noexcept
cdef Py_ssize_t write_float64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float64_t value,
) noexcept
cdef Py_ssize_t write_string_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
) noexcept
cdef Py_ssize_t write_bool_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
) noexcept
cdef Py_ssize_t write_byte_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
) noexcept
cdef Py_ssize_t write_int8_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int8_t[::1] values,
) noexcept
cdef Py_ssize_t write_uint8_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
) noexcept
cdef Py_ssize_t write_int16_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int16_t[::1] values,
) noexcept
cdef Py_ssize_t write_uint16_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint16_t[::1] values,
) noexcept
cdef Py_ssize_t write_int32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int32_t[::1] values,
) noexcept
cdef Py_ssize_t write_uint32_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint32_t[::1] values,
) noexcept
cdef Py_ssize_t write_int64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int64_t[::1] values,
) noexcept
cdef Py_ssize_t write_uint64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint64_t[::1] values,
) noexcept
cdef Py_ssize_t write_float32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
) noexcept
cdef Py_ssize_t write_float64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept
cdef Py_ssize_t write_float64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept
cdef Py_ssize_t write_text_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1
cdef Py_ssize_t write_text_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1

cdef bint read_boolean_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1
cdef uint8_t read_byte_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0
cdef int8_t read_int8_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1
cdef uint8_t read_uint8_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0
cdef int16_t read_int16_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1
cdef uint16_t read_uint16_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0
cdef int32_t read_int32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1
cdef uint32_t read_uint32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0
cdef int64_t read_int64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1
cdef uint64_t read_uint64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0
cdef cnp.float32_t read_float32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1.0
cdef cnp.float64_t read_float64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1.0
cdef bytes read_string_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_bool_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_byte_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_int8_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_uint8_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_int16_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_uint16_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_int32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_uint32_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_int64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_uint64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_float32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_float64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_float64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
)
cdef object read_text_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    int string_collection_mode,
)
cdef object read_text_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    int string_collection_mode,
)
