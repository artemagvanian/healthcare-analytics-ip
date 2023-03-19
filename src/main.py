import json
import sys
from pathlib import Path

from ipinstance import IPInstance
from model_timer import Timer


def main(filepath: str):
    filename = Path(filepath).name
    solver = IPInstance(filepath)

    watch = Timer()
    watch.start()
    result = solver.solve()
    watch.stop()

    print(solver)

    sol_dict = {
        "Instance": filename,
        "Time": str(watch.getElapsed()),
        "Result": str(result),
        "Solution": "OPT"
    }
    print(json.dumps(sol_dict))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
    main(sys.argv[1])
