import subprocess
import sys

result = subprocess.run(
    ["coverage-badge", "-o", "docs/coverage.svg", "-f"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Error generating badge:", result.stderr)
    sys.exit(1)

print("Coverage badge updated at docs/coverage.svg")
