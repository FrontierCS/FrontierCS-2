import json
import sys


def main() -> None:
    unvisited_vulnerability_types = {
        "nested quantifiers",
        "overlapping disjunction",
        "overlapping adjacent quantifiers",
        "starting with large quantifier",
    }
    for entry in map(json.loads, sys.stdin):
        vulnerability_types = entry["vulnerability_types"]
        if len(vulnerability_types) == 1:
            vulnerability_type = vulnerability_types[0]
            if vulnerability_type in unvisited_vulnerability_types:
                print(entry["messages"][-1]["thinking"])
                unvisited_vulnerability_types.remove(vulnerability_type)


if __name__ == "__main__":
    main()
