namespace Voiage.Core;

/// <summary>
/// Value of Information calculations for the voiage core API contract.
/// </summary>
public static class InformationValue
{
    /// <summary>
    /// Calculates Expected Value of Perfect Information through the Rust v1 C
    /// ABI. Rows are samples and columns are strategies.
    /// </summary>
    /// <param name="netBenefits">Net benefit samples by strategy.</param>
    /// <returns>The non-negative EVPI value.</returns>
    /// <exception cref="ArgumentException">Raised when rows are empty or ragged.</exception>
    /// <exception cref="InvalidOperationException">Raised when the Rust ABI is unavailable or rejects the input.</exception>
    public static double Evpi(IReadOnlyList<IReadOnlyList<double>> netBenefits)
    {
        ArgumentNullException.ThrowIfNull(netBenefits);

        if (netBenefits.Count == 0)
        {
            return 0.0;
        }

        var width = netBenefits[0].Count;
        if (width == 0)
        {
            throw new ArgumentException("netBenefits must contain non-empty rows.", nameof(netBenefits));
        }

        var values = new double[netBenefits.Count * width];
        for (var rowIndex = 0; rowIndex < netBenefits.Count; rowIndex++)
        {
            var row = netBenefits[rowIndex];
            if (row.Count != width)
            {
                throw new ArgumentException(
                    "netBenefits rows must have a consistent width.",
                    nameof(netBenefits));
            }

            for (var columnIndex = 0; columnIndex < width; columnIndex++)
            {
                var value = row[columnIndex];
                if (!double.IsFinite(value))
                {
                    throw new ArgumentException(
                        "netBenefits values must be finite numbers.",
                        nameof(netBenefits));
                }
                values[(rowIndex * width) + columnIndex] = value;
            }
        }

        try
        {
            var status = NativeMethods.Evpi(
                values,
                checked((ulong)netBenefits.Count),
                checked((ulong)width),
                out var result);
            if (status != NativeMethods.Ok)
            {
                throw new InvalidOperationException(
                    $"voiage Rust EVPI ABI failed with status {status}.");
            }
            return result;
        }
        catch (DllNotFoundException exception)
        {
            throw new InvalidOperationException(
                "The voiage Rust C ABI library is unavailable.",
                exception);
        }
        catch (EntryPointNotFoundException exception)
        {
            throw new InvalidOperationException(
                "The voiage Rust C ABI does not provide voiage_v1_evpi.",
                exception);
        }
    }
}
