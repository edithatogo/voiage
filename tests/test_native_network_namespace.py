"""Keep local socket tests usable while denying every external build connection."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="Requires the Linux root-owned network namespace qualification runner",
)
def test_native_namespace_allows_loopback_but_denies_external_network(
    tmp_path: Path,
) -> None:
    assert {"VOIAGE_CONTROL_UID", "VOIAGE_CONTROL_GID"} <= os.environ.keys(), (
        "Root qualification requires explicit non-root runner UID/GID"
    )
    uid = int(os.environ["VOIAGE_CONTROL_UID"])
    gid = int(os.environ["VOIAGE_CONTROL_GID"])
    tmp_path.parent.chmod(0o755)
    os.chown(tmp_path, uid, gid)
    tmp_path.chmod(0o750)
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    os.chown(consumer, uid, gid)
    output = tmp_path / "network.json"
    payload = (
        "import errno,json,os,socket;from pathlib import Path;"
        "server=socket.socket();server.settimeout(2);"
        "server.bind(('127.0.0.1',0));server.listen(1);"
        "client=socket.create_connection(server.getsockname(),timeout=2);"
        "peer,address=server.accept();peer.settimeout(2);"
        "client.sendall(b'local');assert peer.recv(5)==b'local';"
        "external=socket.socket();external.settimeout(2);"
        "code=external.connect_ex(('192.0.2.1',443));"
        "assert code==errno.ENETUNREACH;"
        f"assert os.getuid()=={uid} and os.getgid()=={gid};"
        "assert os.getgroups()==[];"
        "assert os.geteuid()==os.getuid() and os.getegid()==os.getgid();"
        "status=Path('/proc/self/status').read_text();"
        "assert int(next(line.split()[1] for line in status.splitlines() if line.startswith('CapEff:')),16)==0;"
        f"Path({str(output)!r}).write_text(json.dumps(dict(loopback=True,external_errno=code,uid=os.getuid())))"
    )
    command = [
        sys.executable,
        "-m",
        "scripts.observe_native_stage",
        "--evidence",
        str(tmp_path / "stage"),
        "--workspace",
        str(tmp_path),
        "--cwd",
        str(consumer),
        "--timeout",
        "12",
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--",
        sys.executable,
        "-c",
        payload,
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    report = json.loads((tmp_path / "stage/resource-report.json").read_text())
    assert result.returncode == 0, (result.stderr, report)
    assert report["cleanup_complete"] is True
    assert json.loads(output.read_text())["loopback"] is True
    assert output.stat().st_uid == uid
    assert output.stat().st_gid == gid


def test_namespace_setup_refuses_the_parent_network(monkeypatch) -> None:
    from scripts.enter_native_namespace import enter_namespace

    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "readlink", lambda path: "net:[host]")
    with pytest.raises(ValueError, match="parent network"):
        enter_namespace("net:[host]", 1001, 1001, ["/bin/true"])


def test_namespace_setup_refuses_external_interfaces(monkeypatch) -> None:
    import socket

    from scripts.enter_native_namespace import enter_namespace

    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(os, "readlink", lambda path: "net:[child]")
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    with pytest.raises(ValueError, match="only loopback"):
        enter_namespace("net:[host]", 1001, 1001, ["/bin/true"])


@pytest.mark.skipif(
    sys.platform == "win32", reason="Requires the Unix account database"
)
def test_build_environment_uses_runner_home_and_drops_credentials(monkeypatch) -> None:
    from types import SimpleNamespace

    from scripts import enter_native_namespace as namespace

    monkeypatch.setattr(
        namespace.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_dir="/home/build", pw_name="build"),
    )
    monkeypatch.setattr(
        namespace.os,
        "environ",
        {
            "HOME": "/root",
            "USER": "root",
            "PATH": "/qualified/bin:/usr/bin",
            "CARGO_NET_OFFLINE": "true",
            "VOIAGE_SOURCE_RECEIPTS": "/receipts",
            "EBROOTGCCCORE": "/prefix/gcc",
            "GITHUB_TOKEN": "synthetic",
            "AWS_SECRET_ACCESS_KEY": "synthetic",
            "GOOGLE_APPLICATION_CREDENTIALS": "/root/auth.json",
            "CARGO_REGISTRIES_CRATES_IO_TOKEN": "synthetic",
        },
    )
    environment = namespace.build_environment(1001)
    assert environment == {
        "HOME": "/home/build",
        "USER": "build",
        "LOGNAME": "build",
        "PATH": "/qualified/bin:/usr/bin",
        "CARGO_NET_OFFLINE": "true",
        "VOIAGE_SOURCE_RECEIPTS": "/receipts",
        "EBROOTGCCCORE": "/prefix/gcc",
    }
