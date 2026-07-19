using System.Text.Json;
using Voiage.Core;

static void AssertEqual(double expected, double actual, string message)
{
    if (Math.Abs(expected - actual) > 1e-12)
    {
        throw new InvalidOperationException($"{message}: expected {expected}, got {actual}");
    }
}

static void AssertWithin(double expected, double actual, double tolerance, string message)
{
    if (Math.Abs(expected - actual) > tolerance)
    {
        throw new InvalidOperationException(
            $"{message}: expected {expected} +/- {tolerance}, got {actual}");
    }
}

static string FindNumericalReference()
{
    DirectoryInfo? directory = new(AppContext.BaseDirectory);
    while (directory is not null)
    {
        string candidate = Path.Combine(
            directory.FullName,
            "specs",
            "numerical-reference",
            "v1",
            "evpi-cases.json");
        if (File.Exists(candidate))
        {
            return candidate;
        }
        directory = directory.Parent;
    }
    throw new FileNotFoundException("Could not locate the shared EVPI reference fixture.");
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

using (JsonDocument reference = JsonDocument.Parse(File.ReadAllText(FindNumericalReference())))
{
    JsonElement root = reference.RootElement;
    if (root.GetProperty("schema_version").GetString() != "1.0.0" ||
        root.GetProperty("method").GetString() != "evpi")
    {
        throw new InvalidOperationException("Unsupported shared EVPI reference fixture.");
    }

    foreach (JsonElement fixtureCase in root.GetProperty("cases").EnumerateArray())
    {
        IReadOnlyList<IReadOnlyList<double>> matrix = fixtureCase
            .GetProperty("net_benefits")
            .EnumerateArray()
            .Select(row => (IReadOnlyList<double>)row
                .EnumerateArray()
                .Select(value => value.GetDouble())
                .ToArray())
            .ToArray();
        JsonElement expected = fixtureCase.GetProperty("expected");
        AssertWithin(
            expected.GetProperty("value").GetDouble(),
            InformationValue.Evpi(matrix),
            expected.GetProperty("atol").GetDouble(),
            $"shared EVPI case {fixtureCase.GetProperty("id").GetString()}");
    }
}

Console.WriteLine("Voiage.Core tests passed.");
