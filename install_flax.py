import subprocess

try:
    subprocess.check_call(["uv", "pip", "install", "flax"])
except subprocess.CalledProcessError as e:
    print(f"Error installing flax: {e}")
