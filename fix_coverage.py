import subprocess
try:
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metamodels.py", "-v"],
        check=True, capture_output=True, text=True
    )
    print("Tests passed.")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    print(e.stdout)
    print(e.stderr)
