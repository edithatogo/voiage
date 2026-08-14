import sys
lines = []
with open("tests/test_metamodels.py") as f:
    lines = f.readlines()
with open("tests/test_metamodels.py", "w") as f:
    for i, line in enumerate(lines):
        f.write(line)
        if "    def test_random_forest_metamodel(sample_data) -> None:" in line:
            pass # Actually we just need to add the happy path assert that is missing?! Wait!
