module Voiage

using Libdl

export evpi

const _OK = Cint(0)

function _ffi_library()
    path = get(ENV, "VOIAGE_FFI_LIBRARY", "libvoiage_ffi")
    return Libdl.dlopen(path)
end

"""
    evpi(net_benefits)

Calculate Expected Value of Perfect Information through the Rust v1 C ABI.
Rows are samples and columns are strategies.
"""
function evpi(net_benefits::AbstractMatrix{<:Real})::Float64
    rows, columns = size(net_benefits)
    if rows == 0 || columns <= 1
        return 0.0
    end
    values = Float64.(net_benefits)
    if any(value -> !isfinite(value), values)
        throw(ArgumentError("net_benefits values must be finite numbers"))
    end

    # Julia is column-major; the ABI consumes row-major values.
    row_major = vec(permutedims(values))
    result = Ref{Cdouble}(0.0)
    handle = _ffi_library()
    try
        function_pointer = Libdl.dlsym(handle, :voiage_v1_evpi)
        status = ccall(
            function_pointer,
            Cint,
            (Ptr{Cdouble}, Culonglong, Culonglong, Ref{Cdouble}),
            row_major,
            Culonglong(rows),
            Culonglong(columns),
            result,
        )
        status == _OK || throw(ErrorException("voiage Rust EVPI ABI failed with status $status"))
        return result[]
    finally
        Libdl.dlclose(handle)
    end
end

end
