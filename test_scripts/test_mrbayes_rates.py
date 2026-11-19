"""Check ModelFinder candidate counts and rates on DNA and protein alignments."""

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import tempfile


CLASSIC = {"E", "I", "G", "I+G"}
EXTENDED = CLASSIC | {"R2", "I+R2"}
ASC = {"ASC", "ASC+G"}


def fixtures(source, directory):
    records = []
    for record in source.read_text().split(">")[1:9]:
        name, *lines = record.splitlines()
        records.append((name, "".join(lines)))
    prefix = directory / source.stem
    complete = prefix.with_suffix(".fa")
    variable = prefix.with_suffix(".variable.fa")
    complete.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))
    sites = [
        i for i, column in enumerate(zip(*(seq for _, seq in records)))
        if len(set(column)) > 1 and all(state.isalpha() for state in column)
    ]
    variable.write_text("".join(
        f">{name}\n{''.join(seq[i] for i in sites)}\n" for name, seq in records
    ))
    return complete, variable


def fit(binary, alignment, datatype, prefix, options):
    command = [
        str(binary), "-s", str(alignment), "-st", datatype, "-T", "1",
        "-seed", "73073", "-v", "-pre", str(prefix), "--score-diff", "all",
        "-cmin", "2", "-cmax", "2", *options,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    count = re.search(r"ModelFinder will test (?:up to )?(\d+) ", result.stdout)
    rows = re.findall(r"^\s*(\d+)\s+(\S+)\s+[-+]?\d+\.\d+\s+\d+\s",
                      result.stdout, re.MULTILINE)
    if result.returncode or not count or not rows:
        raise AssertionError(f"{' '.join(command)}\n{result.stdout}")
    return int(count[1]), rows


def rate_of(model):
    rates = [re.sub(r"^G\d+$", "G", part) for part in model.split("+")[1:]
             if not part.startswith("F")]
    return "+".join(rates) or "E"


def verify_candidates(description, count, rows, expected):
    rates = Counter(rate_of(model) for _, model in rows)
    if (count != sum(expected.values()) or rates != expected
            or len({number for number, _ in rows}) != count):
        raise AssertionError(f"{description}: expected {dict(expected)}, "
                             f"announced {count} candidates and logged {dict(rates)}")
    return rates


def check(binary, baseline, directory):
    data = Path(__file__).resolve().parent / "test_data"
    for datatype, filename, preset_models, all_models, extra in [
        ("DNA", "small.fa", 12, 44, "GTR+R2"),
        ("AA", "turtle_aa.fasta", 18, 56, "LG+R2"),
    ]:
        complete, variable = fixtures(data / filename, directory)
        preset = ["-mset", "mrbayes"]
        # Full tree search avoids the existing DNA nested-model checkpoint
        # assertion when candidate lists contain only ASC models.
        asc_options = ["-mtree"] if datatype == "DNA" else []
        cases = [
            ("mf", complete, ["-m", "MF", *preset], preset_models, CLASSIC, EXTENDED, 0),
            ("mfp", complete, ["-m", "MFP", *preset], preset_models, CLASSIC, EXTENDED, 0),
            ("list", complete, ["-m", "MF", *preset, "-mrate", "E,I,G,I+G"],
             preset_models, CLASSIC, CLASSIC, 0),
            ("auto", complete, ["-m", "MF", *preset, "-mrate", "AUTO"],
             preset_models, EXTENDED, EXTENDED, 0),
            ("all", complete, ["-m", "MF", *preset, "--mrate", "ALL"],
             preset_models, EXTENDED, EXTENDED, 0),
            ("freerate", complete, ["-m", "MF", *preset, "-mrate", "R"],
             preset_models, {"R2"}, {"R2"}, 0),
            ("unrestricted", complete, ["-m", "MF"], all_models, EXTENDED, EXTENDED, 0),
            ("variable", variable, ["-m", "MF", *preset], preset_models, {"E", "G"},
             {"E", "G", "R2", "ASC", "ASC+G", "ASC+R2"} if datatype == "DNA"
             else {"E", "G", "R2"}, 0),
            ("asc", variable, ["-m", "MF+ASC", *preset, *asc_options],
             preset_models, ASC, ASC | {"ASC+R2"}, 0),
            ("asc-list", variable, ["-m", "MF", *preset, "--rates", "ASC,ASC+G", *asc_options],
             preset_models, ASC, ASC, 0),
            ("modifier", complete, ["-m", "MF+R", *preset], preset_models,
             CLASSIC | {"R2"}, EXTENDED, 0),
            ("extra", complete, ["-m", "MF", *preset, "-madd", extra],
             preset_models, CLASSIC, EXTENDED, 1),
        ]
        for label, alignment, options, models, expected, previous, additions in cases:
            description = f"{datatype}/{label}"
            prefix = directory / f"{datatype}-{label}"
            count, rows = fit(binary, alignment, datatype, prefix, options)
            expected_rates = Counter(dict.fromkeys(expected, models))
            if additions:
                expected_rates["R2"] += additions
            rates = verify_candidates(description, count, rows, expected_rates)
            if baseline:
                old_count, old_rows = fit(baseline, alignment, datatype,
                                          prefix.with_suffix(".baseline"), options)
                previous_rates = Counter(dict.fromkeys(previous, models))
                if additions or label == "modifier":
                    previous_rates["R2"] += additions or models
                old_rates = verify_candidates(f"{description} baseline", old_count,
                                              old_rows, previous_rates)
                if not additions and label != "modifier":
                    retained = Counter(model for _, model in old_rows if rate_of(model) in expected)
                    if Counter(model for _, model in rows) != retained:
                        raise AssertionError(f"{description}: retained model identities changed")
                print(f"{description:19} {old_count:3} -> {count:3}  "
                      f"{','.join(sorted(old_rates))} -> {','.join(sorted(rates))}")
            else:
                print(f"{description:19} {count:3}  {','.join(sorted(rates))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--baseline", type=Path, help="optional unpatched IQ-TREE binary")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="iqtree-mrbayes-rates-") as directory:
        check(args.binary.resolve(), args.baseline and args.baseline.resolve(), Path(directory))


if __name__ == "__main__":
    main()
