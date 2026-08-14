module Voiage

using Libdl

export enbs, evpi

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

"""
    enbs(evsi_result, research_cost)

Calculate signed expected net benefit of sampling through the Rust v1 C ABI.
"""
function enbs(evsi_result::Real, research_cost::Real)::Float64
    values = (Float64(evsi_result), Float64(research_cost))
    if any(value -> !isfinite(value), values) || values[2] < 0.0
        throw(ArgumentError("evsi_result must be finite and research_cost must be finite and non-negative"))
    end
    result = Ref{Cdouble}(0.0)
    handle = _ffi_library()
    try
        function_pointer = Libdl.dlsym(handle, :voiage_v1_enbs)
        status = ccall(
            function_pointer,
            Cint,
            (Cdouble, Cdouble, Ref{Cdouble}),
            values[1],
            values[2],
            result,
        )
        status == _OK || throw(ErrorException("voiage Rust ENBS ABI failed with status $status"))
        return result[]
    finally
        Libdl.dlclose(handle)
    end
end

end
