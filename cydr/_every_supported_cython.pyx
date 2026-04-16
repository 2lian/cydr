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
from cpython.mem cimport PyMem_Free, PyMem_Malloc
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
from libc.stdio cimport snprintf
from libc.string cimport memcpy
from libc.string cimport memset

cdef extern from *:
    size_t strnlen(const char* s, size_t maxlen) noexcept nogil

cimport cython
cimport numpy as cnp
import numpy as np


cnp.import_array()


cdef int ENCAPSULATION_HEADER_SIZE = 4
cdef int CDR_ALIGN_MAX = 8
cdef int STRING_COLLECTION_MODE_NUMPY = 0
cdef int STRING_COLLECTION_MODE_LIST = 1
cdef int STRING_COLLECTION_MODE_RAW = 3

cdef class DecodedStringCollection:
    def __cinit__(self):
        self.spans.count = 0
        self.spans.max_size = 0
        self.spans.base = cython.NULL
        self.spans.offsets = NULL
        self.spans.sizes = NULL
        self.owner = None

    def __dealloc__(self):
        if self.spans.offsets != NULL:
            PyMem_Free(self.spans.offsets)
        if self.spans.sizes != NULL:
            PyMem_Free(self.spans.sizes)

    cdef bytes value_at(self, Py_ssize_t index):
        if index < 0:
            index += self.spans.count
        if index < 0 or index >= self.spans.count:
            raise IndexError("DecodedStringCollection index out of range")
        return PyBytes_FromStringAndSize(
            <char*>(self.spans.base + self.spans.offsets[index]),
            self.spans.sizes[index],
        )

    cdef void init(
        self,
        const unsigned char* base,
        object owner,
        Py_ssize_t count,
    ) except *:
        self.spans.count = count
        self.spans.base = base
        self.owner = owner
        if count <= 0:
            self.spans.offsets = NULL
            self.spans.sizes = NULL
            return

        self.spans.offsets = <Py_ssize_t*>PyMem_Malloc(
            count * cython.sizeof(Py_ssize_t)
        )
        if self.spans.offsets == NULL:
            raise MemoryError()

        self.spans.sizes = <Py_ssize_t*>PyMem_Malloc(
            count * cython.sizeof(Py_ssize_t)
        )
        if self.spans.sizes == NULL:
            PyMem_Free(self.spans.offsets)
            self.spans.offsets = NULL
            raise MemoryError()

    cdef void set_item(
        self,
        Py_ssize_t index,
        Py_ssize_t offset,
        Py_ssize_t size,
    ) noexcept:
        self.spans.offsets[index] = offset
        self.spans.sizes[index] = size
        if size > self.spans.max_size:
            self.spans.max_size = size

    cpdef list to_list(self):
        cdef Py_ssize_t index
        cdef list values = []

        for index in range(self.spans.count):
            values.append(self.value_at(index))
        return values

    cpdef cnp.ndarray to_numpy(self):
        cdef Py_ssize_t index
        cdef Py_ssize_t itemsize
        cdef Py_ssize_t sz
        cdef cnp.ndarray values
        cdef char* values_ptr
        cdef char buf[32]
        cdef int npdtype

        # max_size is maintained incrementally in set_item — no scan loop needed.
        itemsize = self.spans.max_size if self.spans.max_size >= 1 else 1

        # Build "S<k>" on the C stack — same idea as C++ ("S"s + to_string(k)).
        # Avoids the Python f-string entirely; np.empty handles the rest.
        npdtype = snprintf(buf, 32, "S%d", <int>itemsize)
        values = np.empty(self.spans.count, dtype=PyBytes_FromStringAndSize(buf, npdtype))
        values_ptr = <char*>cnp.PyArray_DATA(values)
        memset(values_ptr, 0, <size_t>(itemsize * self.spans.count))

        for index in range(self.spans.count):
            span = self.spans.sizes[index]
            memcpy(
                values_ptr + index * itemsize,
                self.spans.base + self.spans.offsets[index],
                <size_t>span,
            )
            # memset(values_ptr + index * itemsize + span, 0, <size_t>(itemsize - span))

        return values

    cpdef object to_final(self, int string_collection_mode):
        if string_collection_mode == STRING_COLLECTION_MODE_NUMPY:
            return self.to_numpy()
        if string_collection_mode == STRING_COLLECTION_MODE_LIST:
            return self.to_list()
        if string_collection_mode == STRING_COLLECTION_MODE_RAW:
            return self
        raise ValueError(
            f"string_collection_mode must be one of "
            f"{STRING_COLLECTION_MODE_NUMPY}, {STRING_COLLECTION_MODE_LIST}, "
            f"or {STRING_COLLECTION_MODE_RAW}; "
            f"got {string_collection_mode!r}."
        )

    def __len__(self):
        return self.spans.count

    def __getitem__(self, index):
        cdef Py_ssize_t start
        cdef Py_ssize_t stop
        cdef Py_ssize_t step
        cdef Py_ssize_t current
        cdef list values

        if isinstance(index, slice):
            start, stop, step = index.indices(self.spans.count)
            values = []
            current = start
            if step > 0:
                while current < stop:
                    values.append(self.value_at(current))
                    current += step
            else:
                while current > stop:
                    values.append(self.value_at(current))
                    current += step
            return values
        return self.value_at(index)

    def __iter__(self):
        cdef Py_ssize_t index
        for index in range(self.spans.count):
            yield self.value_at(index)

    def __contains__(self, value):
        cdef Py_ssize_t index
        cdef bytes candidate

        for index in range(self.spans.count):
            candidate = self.value_at(index)
            if candidate == value:
                return True
        return False

    def __repr__(self):
        return repr(self.to_list())

    def __eq__(self, other):
        if isinstance(other, DecodedStringCollection):
            return self.to_list() == other.to_list()
        return self.to_list() == other

    def __setitem__(self, index, value):
        raise TypeError("DecodedStringCollection is read-only")

    def __delitem__(self, index):
        raise TypeError("DecodedStringCollection is read-only")

cdef inline Py_ssize_t write_string_raw(
    unsigned char* buffer,
    Py_ssize_t pos,
    const char* data,
    Py_ssize_t size,
) noexcept:
    pos = round_up_to_alignment(pos, 4)
    pos = write_uint32_raw(buffer, pos, <uint32_t>(size + 1))
    if size > 0:
        memcpy(buffer + pos, data, <size_t>size)
    buffer[pos + size] = 0
    return pos + size + 1


cdef inline Py_ssize_t advance_string_raw(
    Py_ssize_t pos,
    Py_ssize_t size,
) noexcept:
    pos = round_up_to_alignment(pos, 4)
    return pos + 4 + size + 1


# This prototype assumes the host machine is little-endian.


cdef inline Py_ssize_t round_up_to_alignment(
    Py_ssize_t pos,
    int alignment,
) noexcept:
    cdef int effective_alignment = alignment if alignment <= CDR_ALIGN_MAX else CDR_ALIGN_MAX
    return (pos + effective_alignment - 1) & ~(effective_alignment - 1)


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



cdef inline void require_consumed(
    const unsigned char[::1] data,
    Py_ssize_t pos,
) except *:
    if pos > data.shape[0]:
        raise ValueError("Deserialized past end of buffer.")


cdef inline void validate_encapsulation_header(
    const unsigned char[::1] data,
) except *:
    if ENCAPSULATION_HEADER_SIZE > data.shape[0]:
        raise ValueError("Buffer too short while deserializing.")
    if data[0] != 0 or data[1] != 1 or data[2] != 0 or data[3] != 0:
        raise ValueError("Unsupported encapsulation header.")


cdef inline uint32_t read_uint32_raw_value(
    const unsigned char[::1] data,
    Py_ssize_t pos,
) except? 0:
    cdef uint32_t value = 0
    if pos + <Py_ssize_t>cython.sizeof(uint32_t) > data.shape[0]:
        raise ValueError("Buffer too short while deserializing.")
    memcpy(&value, data_ptr(data) + pos, cython.sizeof(uint32_t))
    return value


cdef inline void read_aligned_value(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    void* out,
    Py_ssize_t itemsize,
    int alignment,
) except *:
    pos[0] = round_up_to_alignment(pos[0], alignment)
    if pos[0] + itemsize > data.shape[0]:
        raise ValueError("Buffer too short while deserializing.")
    memcpy(out, data_ptr(data) + pos[0], <size_t>itemsize)
    pos[0] += itemsize


cdef inline cnp.ndarray read_primitive_array_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
    int type_num,
):
    cdef Py_ssize_t byte_count = count * itemsize
    cdef cnp.npy_intp dims[1]
    cdef cnp.ndarray value

    if count > 0:
        pos[0] = round_up_to_alignment(pos[0], alignment)
        if pos[0] + byte_count > data.shape[0]:
            raise ValueError("Buffer too short while deserializing.")

    dims[0] = <cnp.npy_intp>count
    value = cnp.PyArray_EMPTY(1, dims, type_num, 0)
    if count > 0:
        memcpy(cnp.PyArray_DATA(value), data_ptr(data) + pos[0], <size_t>byte_count)
        pos[0] += byte_count
    return value


cdef inline cnp.ndarray read_primitive_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t itemsize,
    int alignment,
    int type_num,
):
    cdef uint32_t count
    pos[0] = round_up_to_alignment(pos[0], 4)
    count = read_uint32_raw_value(data, pos[0])
    pos[0] += 4
    return read_primitive_array_object(
        data,
        pos,
        count,
        itemsize,
        alignment,
        type_num,
    )


cdef inline bytes read_string_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    cdef uint32_t byte_count
    cdef Py_ssize_t string_size
    cdef bytes value

    pos[0] = round_up_to_alignment(pos[0], 4)
    byte_count = read_uint32_raw_value(data, pos[0])
    pos[0] += 4

    string_size = <Py_ssize_t> byte_count
    if string_size <= 0:
        raise ValueError("Invalid CDR string length.")

    if pos[0] + string_size > data.shape[0]:
        raise ValueError("Buffer too short while deserializing.")
    if data[pos[0] + string_size - 1] != 0:
        raise ValueError("CDR string is missing a trailing NUL byte.")

    # Return owned bytes, not a memoryview. For robotics-style workloads,
    # many short strings are common and copying is faster overall than
    # creating many Python memoryview slice objects. If we revisit large
    # scalar strings later, this is the place to try a zero-copy view again.
    value = PyBytes_FromStringAndSize(<char*> &data[pos[0]], string_size - 1)
    pos[0] += string_size
    return value


cdef inline DecodedStringCollection read_string_array_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    Py_ssize_t count,
):
    cdef Py_ssize_t index
    cdef uint32_t byte_count
    cdef Py_ssize_t string_size
    cdef DecodedStringCollection values = DecodedStringCollection()

    values.init(data_ptr(data), memoryview(data), count)

    for index in range(count):
        pos[0] = round_up_to_alignment(pos[0], 4)
        byte_count = read_uint32_raw_value(data, pos[0])
        pos[0] += 4

        string_size = <Py_ssize_t>byte_count
        if string_size <= 0:
            raise ValueError("Invalid CDR string length.")
        if pos[0] + string_size > data.shape[0]:
            raise ValueError("Buffer too short while deserializing.")
        if data[pos[0] + string_size - 1] != 0:
            raise ValueError("CDR string is missing a trailing NUL byte.")

        values.set_item(index, pos[0], string_size - 1)
        pos[0] += string_size

    return values


cdef inline DecodedStringCollection read_string_sequence_object(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    cdef uint32_t count

    pos[0] = round_up_to_alignment(pos[0], 4)
    count = read_uint32_raw_value(data, pos[0])
    pos[0] += 4

    return read_string_array_object(data, pos, count)


cdef inline Py_ssize_t string_collection_length(object values) except -1:
    if type(values) is list:
        return PyList_GET_SIZE(values)
    return cnp.PyArray_DIM(<cnp.ndarray>values, 0)


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
) noexcept:
    pos = round_up_to_alignment(pos, alignment)
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
) noexcept:
    return write_string_raw(
        buffer, pos, PyBytes_AS_STRING(value), PyBytes_GET_SIZE(value)
    )


cdef inline Py_ssize_t advance_primitive_array_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept:
    if count > 0:
        pos = round_up_to_alignment(pos, alignment)
        pos += count * itemsize
    return pos


cdef inline Py_ssize_t advance_primitive_sequence_position(
    Py_ssize_t pos,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept:
    pos = round_up_to_alignment(pos, 4)
    pos += 4
    return advance_primitive_array_position(pos, count, itemsize, alignment)


cdef inline Py_ssize_t write_primitive_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    const void* data,
    Py_ssize_t count,
    Py_ssize_t itemsize,
    int alignment,
) noexcept:
    cdef Py_ssize_t byte_count
    if count > 0:
        pos = round_up_to_alignment(pos, alignment)
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
) noexcept:
    pos = round_up_to_alignment(pos, 4)
    pos = write_uint32_raw(buffer, pos, <uint32_t> count)
    return write_primitive_array(buffer, pos, data, count, itemsize, alignment)


cdef inline Py_ssize_t advance_string_array_position(
    Py_ssize_t pos,
    object values,
) except -1:
    cdef Py_ssize_t index, n, itemsize
    cdef cnp.ndarray arr
    cdef cnp.dtype descr
    cdef const char* ptr

    if type(values) is list:
        n = PyList_GET_SIZE(values)
        for index in range(n):
            pos = advance_string_raw(
                pos, PyBytes_GET_SIZE(<bytes>PyList_GET_ITEM(values, index))
            )
        return pos

    arr = <cnp.ndarray>values
    n = arr.shape[0]
    descr = <cnp.dtype>cnp.PyArray_DESCR(arr)

    if descr.type_num == cnp.NPY_STRING:
        itemsize = cnp.PyArray_ITEMSIZE(arr)
        ptr = <const char*>cnp.PyArray_DATA(arr)
        for index in range(n):
            pos = advance_string_raw(
                pos, strnlen(ptr + index * itemsize, <size_t>itemsize)
            )
        return pos

    raise TypeError("Expected list[bytes] or np.bytes_ array for string collection.")


cdef inline Py_ssize_t advance_string_sequence_position(
    Py_ssize_t pos,
    object values,
) except -1:
    pos = round_up_to_alignment(pos, 4)
    pos += 4
    return advance_string_array_position(pos, values)


cdef inline Py_ssize_t write_string_array(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1:
    cdef Py_ssize_t index, n, itemsize
    cdef cnp.ndarray arr
    cdef cnp.dtype descr
    cdef const char* ptr

    if type(values) is list:
        n = PyList_GET_SIZE(values)
        for index in range(n):
            pos = write_string_raw(
                buffer, pos,
                PyBytes_AS_STRING(<bytes>PyList_GET_ITEM(values, index)),
                PyBytes_GET_SIZE(<bytes>PyList_GET_ITEM(values, index)),
            )
        return pos

    arr = <cnp.ndarray>values
    n = arr.shape[0]
    descr = <cnp.dtype>cnp.PyArray_DESCR(arr)

    if descr.type_num == cnp.NPY_STRING:
        itemsize = cnp.PyArray_ITEMSIZE(arr)
        ptr = <const char*>cnp.PyArray_DATA(arr)
        for index in range(n):
            pos = write_string_raw(
                buffer, pos,
                ptr + index * itemsize,
                strnlen(ptr + index * itemsize, <size_t>itemsize),
            )
        return pos

    raise TypeError("Expected list[bytes] or np.bytes_ array for string collection.")


cdef inline Py_ssize_t write_string_sequence(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1:
    cdef Py_ssize_t count = string_collection_length(values)
    pos = round_up_to_alignment(pos, 4)
    pos = write_uint32_raw(buffer, pos, <uint32_t>count)
    return write_string_array(buffer, pos, values)


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
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t write_int8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int8_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int8_t), 1)


cdef inline Py_ssize_t write_uint8_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint8_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t write_int16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int16_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int16_t), 2)


cdef inline Py_ssize_t write_uint16_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint16_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint16_t), 2)


cdef inline Py_ssize_t write_int32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int32_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int32_t), 4)


cdef inline Py_ssize_t write_uint32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint32_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint32_t), 4)


cdef inline Py_ssize_t write_int64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    int64_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(int64_t), 8)


cdef inline Py_ssize_t write_uint64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    uint64_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(uint64_t), 8)


cdef inline Py_ssize_t write_float32_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float32_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(cnp.float32_t), 4)


cdef inline Py_ssize_t write_float64_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    cnp.float64_t value,
) noexcept:
    return write_aligned_value(buffer, pos, &value, cython.sizeof(cnp.float64_t), 8)


cdef inline Py_ssize_t write_string_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    bytes value,
) noexcept:
    return write_string(buffer, pos, value)


cdef inline bint read_boolean_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1:
    if pos[0] + 1 > data.shape[0]:
        raise ValueError("Buffer too short while deserializing.")
    cdef bint value = data[pos[0]] != 0
    pos[0] += 1
    return value


cdef inline uint8_t read_byte_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0:
    cdef uint8_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(uint8_t), 1)
    return value


cdef inline int8_t read_int8_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1:
    cdef int8_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(int8_t), 1)
    return value


cdef inline uint8_t read_uint8_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0:
    cdef uint8_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(uint8_t), 1)
    return value


cdef inline int16_t read_int16_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1:
    cdef int16_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(int16_t), 2)
    return value


cdef inline uint16_t read_uint16_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0:
    cdef uint16_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(uint16_t), 2)
    return value


cdef inline int32_t read_int32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1:
    cdef int32_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(int32_t), 4)
    return value


cdef inline uint32_t read_uint32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0:
    cdef uint32_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(uint32_t), 4)
    return value


cdef inline int64_t read_int64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1:
    cdef int64_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(int64_t), 8)
    return value


cdef inline uint64_t read_uint64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? 0:
    cdef uint64_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(uint64_t), 8)
    return value


cdef inline cnp.float32_t read_float32_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1.0:
    cdef cnp.float32_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(cnp.float32_t), 4)
    return value


cdef inline cnp.float64_t read_float64_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
) except? -1.0:
    cdef cnp.float64_t value = 0
    read_aligned_value(data, pos, &value, cython.sizeof(cnp.float64_t), 8)
    return value


cdef inline bytes read_string_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_string_object(data, pos)


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
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        bool_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.npy_bool),
        1,
    )


cdef inline Py_ssize_t write_byte_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint8_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint8_t),
        1,
    )


cdef inline Py_ssize_t write_int8_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int8_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int8_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int8_t),
        1,
    )


cdef inline Py_ssize_t write_uint8_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint8_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint8_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint8_t),
        1,
    )


cdef inline Py_ssize_t write_int16_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int16_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int16_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int16_t),
        2,
    )


cdef inline Py_ssize_t write_uint16_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint16_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint16_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint16_t),
        2,
    )


cdef inline Py_ssize_t write_int32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int32_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int32_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int32_t),
        4,
    )


cdef inline Py_ssize_t write_uint32_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint32_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint32_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint32_t),
        4,
    )


cdef inline Py_ssize_t write_int64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const int64_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        int64_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(int64_t),
        8,
    )


cdef inline Py_ssize_t write_uint64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const uint64_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        uint64_view_ptr(values),
        values.shape[0],
        cython.sizeof(uint64_t),
        8,
    )


cdef inline Py_ssize_t write_float32_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        float32_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float32_t),
        4,
    )


cdef inline Py_ssize_t write_float64_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept:
    return write_primitive_array(
        buffer,
        pos,
        float64_view_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float64_t),
        8,
    )


cdef inline Py_ssize_t write_float64_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept:
    return write_primitive_sequence(
        buffer,
        pos,
        float64_sequence_ptr(values),
        values.shape[0],
        cython.sizeof(cnp.float64_t),
        8,
    )


cdef inline Py_ssize_t write_text_array_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1:
    require_fixed_length(string_collection_length(values), 2, "text_array")
    return write_string_array(buffer, pos, values)


cdef inline Py_ssize_t write_text_sequence_field(
    unsigned char* buffer,
    Py_ssize_t pos,
    object values,
) except -1:
    return write_string_sequence(buffer, pos, values)


cdef inline cnp.ndarray read_bool_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.npy_bool),
        1,
        cnp.NPY_BOOL,
    )


cdef inline cnp.ndarray read_byte_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        3,
        cython.sizeof(uint8_t),
        1,
        cnp.NPY_UINT8,
    )


cdef inline cnp.ndarray read_int8_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int8_t),
        1,
        cnp.NPY_INT8,
    )


cdef inline cnp.ndarray read_uint8_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        3,
        cython.sizeof(uint8_t),
        1,
        cnp.NPY_UINT8,
    )


cdef inline cnp.ndarray read_int16_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int16_t),
        2,
        cnp.NPY_INT16,
    )


cdef inline cnp.ndarray read_uint16_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint16_t),
        2,
        cnp.NPY_UINT16,
    )


cdef inline cnp.ndarray read_int32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int32_t),
        4,
        cnp.NPY_INT32,
    )


cdef inline cnp.ndarray read_uint32_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint32_t),
        4,
        cnp.NPY_UINT32,
    )


cdef inline cnp.ndarray read_int64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(int64_t),
        8,
        cnp.NPY_INT64,
    )


cdef inline cnp.ndarray read_uint64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(uint64_t),
        8,
        cnp.NPY_UINT64,
    )


cdef inline cnp.ndarray read_float32_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.float32_t),
        4,
        cnp.NPY_FLOAT32,
    )


cdef inline cnp.ndarray read_float64_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_array_object(
        data,
        pos,
        2,
        cython.sizeof(cnp.float64_t),
        8,
        cnp.NPY_FLOAT64,
    )


cdef inline cnp.ndarray read_float64_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
):
    return read_primitive_sequence_object(
        data,
        pos,
        cython.sizeof(cnp.float64_t),
        8,
        cnp.NPY_FLOAT64,
    )


cdef inline object read_text_array_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    int string_collection_mode,
):
    return read_string_array_object(data, pos, 2).to_final(string_collection_mode)


cdef inline object read_text_sequence_field(
    const unsigned char[::1] data,
    Py_ssize_t* pos,
    int string_collection_mode,
):
    return read_string_sequence_object(data, pos).to_final(string_collection_mode)


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
) noexcept:
    return advance_primitive_array_position(pos, 1, itemsize, alignment)


cdef inline Py_ssize_t advance_string_field(
    Py_ssize_t pos,
    bytes value,
) noexcept:
    pos = round_up_to_alignment(pos, 4)
    return pos + 4 + bytes_size(value) + 1


cdef inline Py_ssize_t advance_boolean_field(Py_ssize_t pos) noexcept:
    return pos + 1


cdef inline Py_ssize_t advance_byte_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t advance_int8_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int8_t), 1)


cdef inline Py_ssize_t advance_uint8_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t advance_int16_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int16_t), 2)


cdef inline Py_ssize_t advance_uint16_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint16_t), 2)


cdef inline Py_ssize_t advance_int32_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int32_t), 4)


cdef inline Py_ssize_t advance_uint32_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint32_t), 4)


cdef inline Py_ssize_t advance_int64_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(int64_t), 8)


cdef inline Py_ssize_t advance_uint64_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(uint64_t), 8)


cdef inline Py_ssize_t advance_float32_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(cnp.float32_t), 4)


cdef inline Py_ssize_t advance_float64_field(
    Py_ssize_t pos,
) noexcept:
    return advance_scalar_field(pos, cython.sizeof(cnp.float64_t), 8)


cdef inline Py_ssize_t advance_bool_sequence_field(
    Py_ssize_t pos,
    const cnp.npy_bool[::1] values,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.npy_bool), 1
    )


cdef inline Py_ssize_t advance_byte_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 3, "byte_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t advance_int8_sequence_field(
    Py_ssize_t pos,
    const int8_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int8_t), 1)


cdef inline Py_ssize_t advance_uint8_array_field(
    Py_ssize_t pos,
    const uint8_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 3, "uint8_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint8_t), 1)


cdef inline Py_ssize_t advance_int16_sequence_field(
    Py_ssize_t pos,
    const int16_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int16_t), 2)


cdef inline Py_ssize_t advance_uint16_array_field(
    Py_ssize_t pos,
    const uint16_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint16_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint16_t), 2)


cdef inline Py_ssize_t advance_int32_sequence_field(
    Py_ssize_t pos,
    const int32_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int32_t), 4)


cdef inline Py_ssize_t advance_uint32_array_field(
    Py_ssize_t pos,
    const uint32_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint32_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint32_t), 4)


cdef inline Py_ssize_t advance_int64_sequence_field(
    Py_ssize_t pos,
    const int64_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(pos, values.shape[0], cython.sizeof(int64_t), 8)


cdef inline Py_ssize_t advance_uint64_array_field(
    Py_ssize_t pos,
    const uint64_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 2, "uint64_array")
    return advance_primitive_array_position(pos, values.shape[0], cython.sizeof(uint64_t), 8)


cdef inline Py_ssize_t advance_float32_sequence_field(
    Py_ssize_t pos,
    const cnp.float32_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.float32_t), 4
    )


cdef inline Py_ssize_t advance_float64_array_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) except -1:
    require_fixed_length(values.shape[0], 2, "float64_array")
    return advance_primitive_array_position(
        pos, values.shape[0], cython.sizeof(cnp.float64_t), 8
    )


cdef inline Py_ssize_t advance_float64_sequence_field(
    Py_ssize_t pos,
    const cnp.float64_t[::1] values,
) noexcept:
    return advance_primitive_sequence_position(
        pos, values.shape[0], cython.sizeof(cnp.float64_t), 8
    )


cdef inline Py_ssize_t advance_text_array_field(
    Py_ssize_t pos,
    object values,
) except -1:
    require_fixed_length(string_collection_length(values), 2, "text_array")
    return advance_string_array_position(pos, values)


cdef inline Py_ssize_t advance_text_sequence_field(
    Py_ssize_t pos,
    object values,
) except -1:
    return advance_string_sequence_position(pos, values)


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
    object text_array,
    object text_sequence,
) except -1:
    cdef Py_ssize_t pos = 0

    pos = advance_boolean_field(pos)
    pos = advance_byte_field(pos)
    pos = advance_int8_field(pos)
    pos = advance_uint8_field(pos)
    pos = advance_int16_field(pos)
    pos = advance_uint16_field(pos)
    pos = advance_int32_field(pos)
    pos = advance_uint32_field(pos)
    pos = advance_int64_field(pos)
    pos = advance_uint64_field(pos)
    pos = advance_float32_field(pos)
    pos = advance_float64_field(pos)
    pos = advance_string_field(pos, text)
    pos = advance_int32_field(pos)
    pos = advance_uint32_field(pos)
    pos = advance_string_field(pos, header_frame_id)

    pos = advance_bool_sequence_field(pos, bool_sequence)
    pos = advance_byte_array_field(pos, byte_array)
    pos = advance_int8_sequence_field(pos, int8_sequence)
    pos = advance_uint8_array_field(pos, uint8_array)
    pos = advance_int16_sequence_field(pos, int16_sequence)
    pos = advance_uint16_array_field(pos, uint16_array)
    pos = advance_int32_sequence_field(pos, int32_sequence)
    pos = advance_uint32_array_field(pos, uint32_array)
    pos = advance_int64_sequence_field(pos, int64_sequence)
    pos = advance_uint64_array_field(pos, uint64_array)
    pos = advance_float32_sequence_field(pos, float32_sequence)
    pos = advance_float64_array_field(pos, float64_array)
    pos = advance_text_array_field(pos, text_array)
    pos = advance_text_sequence_field(pos, text_sequence)
    return pos + ENCAPSULATION_HEADER_SIZE


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
    object text_array,
    object text_sequence,
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
    cdef unsigned char* payload = buffer + ENCAPSULATION_HEADER_SIZE
    cdef Py_ssize_t pos = 0

    write_encapsulation_header(buffer)

    pos = write_boolean_field(payload, pos, boolean_value)
    pos = write_byte_field(payload, pos, byte_value)
    pos = write_int8_field(payload, pos, signed_int8)
    pos = write_uint8_field(payload, pos, unsigned_int8)
    pos = write_int16_field(payload, pos, signed_int16)
    pos = write_uint16_field(payload, pos, unsigned_int16)
    pos = write_int32_field(payload, pos, signed_int32)
    pos = write_uint32_field(payload, pos, unsigned_int32)
    pos = write_int64_field(payload, pos, signed_int64)
    pos = write_uint64_field(payload, pos, unsigned_int64)
    pos = write_float32_field(payload, pos, float32_value)
    pos = write_float64_field(payload, pos, float64_value)
    pos = write_string_field(payload, pos, text)
    pos = write_int32_field(payload, pos, header_stamp_sec)
    pos = write_uint32_field(payload, pos, header_stamp_nanosec)
    pos = write_string_field(payload, pos, header_frame_id)

    pos = write_bool_sequence_field(payload, pos, bool_sequence)
    pos = write_byte_array_field(payload, pos, byte_array)
    pos = write_int8_sequence_field(payload, pos, int8_sequence)
    pos = write_uint8_array_field(payload, pos, uint8_array)
    pos = write_int16_sequence_field(payload, pos, int16_sequence)
    pos = write_uint16_array_field(payload, pos, uint16_array)
    pos = write_int32_sequence_field(payload, pos, int32_sequence)
    pos = write_uint32_array_field(payload, pos, uint32_array)
    pos = write_int64_sequence_field(payload, pos, int64_sequence)
    pos = write_uint64_array_field(payload, pos, uint64_array)
    pos = write_float32_sequence_field(payload, pos, float32_sequence)
    pos = write_float64_array_field(payload, pos, float64_array)
    pos = write_text_array_field(payload, pos, text_array)
    pos = write_text_sequence_field(payload, pos, text_sequence)
    return output


cpdef dict deserialize_every_supported_schema(
    const unsigned char[::1] data,
    int string_collection_mode=STRING_COLLECTION_MODE_NUMPY,
):
    cdef const unsigned char[::1] payload
    cdef Py_ssize_t pos = 0
    cdef bint boolean_value
    cdef uint8_t byte_value
    cdef int8_t signed_int8
    cdef uint8_t unsigned_int8
    cdef int16_t signed_int16
    cdef uint16_t unsigned_int16
    cdef int32_t signed_int32
    cdef uint32_t unsigned_int32
    cdef int64_t signed_int64
    cdef uint64_t unsigned_int64
    cdef cnp.float32_t float32_value
    cdef cnp.float64_t float64_value
    cdef bytes text
    cdef int32_t header_stamp_sec
    cdef uint32_t header_stamp_nanosec
    cdef bytes header_frame_id
    cdef cnp.ndarray bool_sequence
    cdef cnp.ndarray byte_array
    cdef cnp.ndarray int8_sequence
    cdef cnp.ndarray uint8_array
    cdef cnp.ndarray int16_sequence
    cdef cnp.ndarray uint16_array
    cdef cnp.ndarray int32_sequence
    cdef cnp.ndarray uint32_array
    cdef cnp.ndarray int64_sequence
    cdef cnp.ndarray uint64_array
    cdef cnp.ndarray float32_sequence
    cdef cnp.ndarray float64_array
    cdef object text_array
    cdef object text_sequence

    validate_encapsulation_header(data)
    payload = data[ENCAPSULATION_HEADER_SIZE:]

    boolean_value = read_boolean_field(payload, &pos)
    byte_value = read_byte_field(payload, &pos)
    signed_int8 = read_int8_field(payload, &pos)
    unsigned_int8 = read_uint8_field(payload, &pos)
    signed_int16 = read_int16_field(payload, &pos)
    unsigned_int16 = read_uint16_field(payload, &pos)
    signed_int32 = read_int32_field(payload, &pos)
    unsigned_int32 = read_uint32_field(payload, &pos)
    signed_int64 = read_int64_field(payload, &pos)
    unsigned_int64 = read_uint64_field(payload, &pos)
    float32_value = read_float32_field(payload, &pos)
    float64_value = read_float64_field(payload, &pos)
    text = read_string_field(payload, &pos)
    header_stamp_sec = read_int32_field(payload, &pos)
    header_stamp_nanosec = read_uint32_field(payload, &pos)
    header_frame_id = read_string_field(payload, &pos)
    bool_sequence = read_bool_sequence_field(payload, &pos)
    byte_array = read_byte_array_field(payload, &pos)
    int8_sequence = read_int8_sequence_field(payload, &pos)
    uint8_array = read_uint8_array_field(payload, &pos)
    int16_sequence = read_int16_sequence_field(payload, &pos)
    uint16_array = read_uint16_array_field(payload, &pos)
    int32_sequence = read_int32_sequence_field(payload, &pos)
    uint32_array = read_uint32_array_field(payload, &pos)
    int64_sequence = read_int64_sequence_field(payload, &pos)
    uint64_array = read_uint64_array_field(payload, &pos)
    float32_sequence = read_float32_sequence_field(payload, &pos)
    float64_array = read_float64_array_field(payload, &pos)
    text_array = read_text_array_field(payload, &pos, string_collection_mode)
    text_sequence = read_text_sequence_field(payload, &pos, string_collection_mode)
    require_consumed(payload, pos)

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
