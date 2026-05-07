namespace Voiage.Core;

/// <summary>
/// Value of Information calculations for the voiage core API contract.
/// </summary>
public static class InformationValue
{
    /// <summary>
    /// Calculates Expected Value of Perfect Information from a net-benefit matrix.
    /// Rows are samples and columns are strategies.
    /// </summary>
    /// <param name="netBenefits">Net benefit samples by strategy.</param>
    /// <returns>The non-negative EVPI value.</returns>
    /// <exception cref="ArgumentException">Raised when rows are empty or ragged.</exception>
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

        var strategySums = new double[width];
        var maxSum = 0.0;

        foreach (var row in netBenefits)
        {
            if (row.Count != width)
            {
                throw new ArgumentException(
                    "netBenefits rows must have a consistent width.",
                    nameof(netBenefits));
            }

            var rowMax = row[0];
            for (var index = 0; index < row.Count; index++)
            {
                var value = row[index];
                strategySums[index] += value;
                if (value > rowMax)
                {
                    rowMax = value;
                }
            }

            maxSum += rowMax;
        }

        if (width <= 1)
        {
            return 0.0;
        }

        var sampleCount = netBenefits.Count;
        var maxExpected = strategySums.Select(sum => sum / sampleCount).Max();
        var expectedPerfectInformation = maxSum / sampleCount;
        return Math.Max(0.0, expectedPerfectInformation - maxExpected);
    }
}
