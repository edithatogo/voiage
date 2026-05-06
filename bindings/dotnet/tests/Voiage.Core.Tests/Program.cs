using Voiage.Core;

static void AssertEqual(double expected, double actual, string message)
{
    if (Math.Abs(expected - actual) > 1e-12)
    {
        throw new InvalidOperationException($"{message}: expected {expected}, got {actual}");
    }
}

AssertEqual(
    3.0,
    InformationValue.Evpi(
        [
            [10.0, 1.0],
            [2.0, 8.0],
        ]),
    "EVPI simple matrix");

AssertEqual(
    0.0,
    InformationValue.Evpi([]),
    "EVPI empty matrix");

try
{
    InformationValue.Evpi(
        [
            [1.0],
            [1.0, 2.0],
        ]);
    throw new InvalidOperationException("Ragged rows should fail.");
}
catch (ArgumentException)
{
    // Expected.
}

Console.WriteLine("Voiage.Core tests passed.");
