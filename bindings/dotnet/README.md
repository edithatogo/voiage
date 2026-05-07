# Voiage.Core

.NET 11 NuGet package scaffold for the voiage core API contract.

## Setup

From `bindings/dotnet/`:

```bash
dotnet build src/Voiage.Core/Voiage.Core.csproj
dotnet run --project tests/Voiage.Core.Tests/Voiage.Core.Tests.csproj
dotnet pack src/Voiage.Core/Voiage.Core.csproj --configuration Release /p:Version=<tag-version> /p:PackageVersion=<tag-version>
```

## First workflow

```csharp
using Voiage.Core;

var netBenefits = new double[][]
{
    new[] { 10.0, 1.0 },
    new[] { 2.0, 8.0 },
};

var evpi = InformationValue.Evpi(netBenefits);
Console.WriteLine($"EVPI: {evpi:0.0}");
```

This prints `EVPI: 3.0` for the canonical two-strategy matrix.

## Release and caveats

The package targets `net11.0` and is intended for NuGet publishing from the
polyglot release workflow when `NUGET_API_KEY` is configured. Release jobs are
tagged `dotnet-v*`, build and run the console test harness, pack the NuGet
artifact with the tag-derived version, and attach both the `.nupkg` and a
source archive to the GitHub release. The .NET layer should stay a thin adapter
over the Rust core rather than a second policy engine.

You need a .NET SDK that supports `net11.0` to build and test this package
locally; older SDKs will fail before the walkthrough can run.
