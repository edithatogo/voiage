module Voiage

using Libdl
using Arrow
using JSON
using Tables

export evpi, normalize_decision_problem, normalize_statistical_assurance, read_voiage_arrow

const _OK = Cint(0)

function _ffi_library()
    path = get(ENV, "VOIAGE_FFI_LIBRARY", "libvoiage_ffi")
    return Libdl.dlopen(path)
end

function _normalize_json(symbol::Symbol, value)
    encoded = value isa AbstractString ? String(value) : JSON.json(value)
    input = Vector{UInt8}(codeunits(encoded))
    required = Ref{Culonglong}(0)
    handle = _ffi_library()
    try
        function_pointer = Libdl.dlsym(handle, symbol)
        status = ccall(
            function_pointer,
            Cint,
            (Ptr{UInt8}, Culonglong, Ptr{UInt8}, Culonglong, Ref{Culonglong}),
            input,
            Culonglong(length(input)),
            Ptr{UInt8}(C_NULL),
            Culonglong(0),
            required,
        )
        status == _OK || throw(ArgumentError("voiage Rust JSON ABI rejected the input (status $status)"))
        required[] > 1 || throw(ErrorException("voiage Rust JSON ABI returned an invalid size"))
        output = Vector{UInt8}(undef, Int(required[]))
        status = ccall(
            function_pointer,
            Cint,
            (Ptr{UInt8}, Culonglong, Ptr{UInt8}, Culonglong, Ref{Culonglong}),
            input,
            Culonglong(length(input)),
            output,
            Culonglong(length(output)),
            required,
        )
        status == _OK || throw(ErrorException("voiage Rust JSON ABI copy failed with status $status"))
        output[end] == 0 || throw(ErrorException("voiage Rust JSON ABI result is not NUL terminated"))
        return JSON.parse(String(output[1:end-1]))
    finally
        Libdl.dlclose(handle)
    end
end

"""
    normalize_decision_problem(problem)

Validate and normalize a v1 Decision Problem through the Rust C ABI.
"""
normalize_decision_problem(problem) =
    _normalize_json(:voiage_v1_decision_problem_json, problem)

"""
    normalize_statistical_assurance(assurance)

Validate and normalize a v1 statistical-assurance envelope through Rust.
"""
normalize_statistical_assurance(assurance) =
    _normalize_json(:voiage_v1_statistical_assurance_json, assurance)

"""
    read_voiage_arrow(path)

Read a canonical Arrow IPC table and require its lossless `payload_json`
contract column.
"""
function read_voiage_arrow(path::AbstractString)
    table = Arrow.Table(path)
    columns = Tuple(Tables.columnnames(table))
    decision_columns = (
        :decision_problem_id,
        :title,
        :analysis_type,
        :currency,
        :willingness_to_pay,
        :outcome_names,
        :intervention_count,
        :payload_json,
    )
    assurance_columns = (
        :reporting_class,
        :replications,
        :stopping_reason,
        :has_confidence_interval,
        :has_convergence_evidence,
        :has_rng_identity,
        :payload_json,
    )
    columns in (decision_columns, assurance_columns) ||
        throw(ArgumentError("Arrow table does not match a pinned voiage v1 schema"))
    return table
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
