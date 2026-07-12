from pathlib import Path


def test_dotnet_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert "dotnet-v*" in workflow_text
    assert (
        "dotnet build src/Voiage.Core/Voiage.Core.csproj --configuration Release"
        in workflow_text
    )
    assert (
        "dotnet pack src/Voiage.Core/Voiage.Core.csproj --configuration Release --no-build --output nupkg"
        in workflow_text
    )
    assert 'dotnet nuget push "nupkg/*.nupkg"' in workflow_text
    assert "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65" in workflow_text

    assert (
        "The .NET binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "NuGet publication is automated on `dotnet-v*` tags when credentials are present."
        in checklist_text
    )
