import subprocess
import sys
from anybadge import Badge


def coverage_badge():
    """Generate a coverage badge using the coverage-badge tool."""
    result = subprocess.run(
        ["coverage-badge", "-o", "docs/coverage.svg", "-f"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error generating badge:", result.stderr)
        sys.exit(1)

    print("Coverage badge updated at docs/coverage.svg")


def coverage_flake8():
    # 1. Run flake8 and capture the output
    # We use --count to get the total number of errors
    # on the last line of stdout/stderr
    result = subprocess.run(
        ['flake8', '.', '--count'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Flake8 returns the count on stdout (or 0 if it passes perfectly)
    # If there are errors, the count is usually the last stripped line
    output = result.stdout.strip().split('\n')[-1]

    try:
        error_count = int(output)
    except ValueError:
        # If flake8 finds 0 errors, --count might return
        # nothing or 0 depending on version
        error_count = 0

    # 2. Define badge thresholds and colors
    thresholds = {10: 'green', 20: 'yellow', 30: 'orange', 50: 'red'}

    badge = Badge(
        label='flake8',
        value=error_count,
        thresholds=thresholds,
        default_color='red'
    )

    # 3. Write the badge to a file
    badge.write_badge('docs/flake8.svg', overwrite=True)
    print(f"Flake8 badge updated successfully! Total errors: {error_count}")


if __name__ == "__main__":
    coverage_badge()
    coverage_flake8()
