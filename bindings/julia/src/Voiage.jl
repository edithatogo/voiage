module Voiage

using Statistics

export evpi

"""
    evpi(net_benefits)

Calculate Expected Value of Perfect Information from a net-benefit matrix.
Rows are samples and columns are strategies.
"""
function evpi(net_benefits::AbstractMatrix{<:Real})::Float64
    if isempty(net_benefits) || size(net_benefits, 2) <= 1
        return 0.0
    end

    expected_by_strategy = vec(mean(net_benefits; dims = 1))
    expected_current_value = maximum(expected_by_strategy)
    expected_perfect_information = mean(maximum(net_benefits; dims = 2))

    return max(0.0, expected_perfect_information - expected_current_value)
end

end
