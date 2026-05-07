from __future__ import annotations

import pytest

from voiage import cli


def test_log_cli_debug_emits_without_and_with_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, tuple[object, ...]]] = []

    def fake_debug(message: str, *args: object) -> None:
        messages.append((message, args))

    monkeypatch.setitem(cli._CLI_STATE, "verbose", True)
    monkeypatch.setattr(cli._CLI_LOGGER, "debug", fake_debug)

    cli._log_cli_debug("calculate-evpi")
    cli._log_cli_debug("calculate-evppi", label="x", value=2)

    assert messages == [
        ("%s", ("calculate-evpi",)),
        ("%s: %s", ("calculate-evppi", "label='x', value=2")),
    ]


def test_log_cli_debug_is_silent_when_not_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_debug(message: str, *args: object) -> None:
        nonlocal called
        called = True
        _ = message, args

    monkeypatch.setitem(cli._CLI_STATE, "verbose", False)
    monkeypatch.setattr(cli._CLI_LOGGER, "debug", fake_debug)

    cli._log_cli_debug("calculate-evpi", label="ignored")

    assert called is False


def test_format_output_supports_all_output_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"beta": 2, "alpha": {"nested": True}}

    monkeypatch.setitem(cli._CLI_STATE, "output_format", "text")
    assert cli._format_output("plain text", payload) == "plain text"

    monkeypatch.setitem(cli._CLI_STATE, "output_format", "json")
    json_output = cli._format_output("plain text", payload)
    assert '"alpha": {' in json_output
    assert '"beta": 2' in json_output

    monkeypatch.setitem(cli._CLI_STATE, "output_format", "csv")
    csv_output = cli._format_output("plain text", payload)
    assert csv_output.splitlines()[0] == "beta,alpha"
    assert csv_output.splitlines()[1] == '2,"{""nested"": true}"'


def test_format_output_rejects_invalid_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(cli._CLI_STATE, "output_format", "xml")

    with pytest.raises(ValueError, match="Unsupported output format"):
        cli._format_output("plain text", {"command": "calculate-evpi"})


@pytest.mark.parametrize(
    ("quiet", "format_name", "expected"),
    [
        (False, "text", True),
        (True, "text", False),
        (False, "json", False),
        (False, "csv", False),
    ],
)
def test_should_echo_status_messages(
    monkeypatch: pytest.MonkeyPatch,
    quiet: bool,
    format_name: str,
    expected: bool,
) -> None:
    monkeypatch.setitem(cli._CLI_STATE, "quiet", quiet)
    monkeypatch.setitem(cli._CLI_STATE, "output_format", format_name)

    assert cli._should_echo_status_messages() is expected


def test_read_scalar_input_accepts_numeric_string() -> None:
    assert cli._read_scalar_input("12.5", "EVSI") == pytest.approx(12.5)


def test_read_scalar_input_reads_numeric_file(tmp_path) -> None:
    scalar_file = tmp_path / "evsi.txt"
    scalar_file.write_text("EVSI: 7.25\n", encoding="utf-8")

    assert cli._read_scalar_input(str(scalar_file), "EVSI") == pytest.approx(7.25)


def test_read_scalar_input_rejects_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="EVSI file not found"):
        cli._read_scalar_input(str(missing), "EVSI")


def test_read_scalar_input_rejects_non_numeric_file(tmp_path) -> None:
    scalar_file = tmp_path / "evsi.txt"
    scalar_file.write_text("not a number", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a numeric value"):
        cli._read_scalar_input(str(scalar_file), "EVSI")


def test_optional_float_field_handles_present_missing_and_invalid_values() -> None:
    payload = {"population": 100.0}

    assert cli._optional_float_field(payload, "population", None) == pytest.approx(
        100.0
    )
    assert cli._optional_float_field({}, "population", None) is None

    with pytest.raises(TypeError, match="population must be a number"):
        cli._optional_float_field({"population": "100"}, "population", None)
