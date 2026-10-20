"""Get RegexScalpel results"""

import itertools
import logging
import subprocess
import sys


def main():
    for pattern in sys.stdin:
        try:
            cmd = ["java", "-jar", "third-party/ReScue/release/ReScue.jar"]
            completed_process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=60,
                stdin=pattern.encode("utf-8"),
            )
            output = completed_process.stdout.decode("utf-8")
            print(
                output.splitlines()[-2],
                flush=True,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning(e)
            pass


if __name__ == "__main__":
    main()
