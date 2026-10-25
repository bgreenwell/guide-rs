"""Small importance cases with strong signals and independent raw-score checks.

Native normalized magnitudes are not compared numerically across different RNGs.
The tests compare important sets and identifiable rank relations instead.
"""
from reference_cases import case, cases as fitting_cases


def cases():
    result = {}
    def add(value, name, statistics, important, zero):
        value["name"] = name
        value["importance_options"] = {"permutations": 300, "max_depth": 1, "seed": 0}
        value["expected_raw_statistics"] = statistics
        value["expected_important"] = important
        value["expected_zero"] = zero
        result[name] = value
    def statistic(x, df):
        return {"chi_square": x, "degrees_of_freedom": df}
    for classification in [False, True]:
        prefix = "class" if classification else "ls"
        def y(condition):
            return ("b" if condition else "a") if classification else (8 if condition else 2)
        value = case("unused", classification,
                     [(n, "numeric") for n in ["signal", "copy", "noise", "constant"]],
                     [[i // 40, i // 40, i % 2, 1] for i in range(80)],
                     [y(i >= 40) for i in range(80)], depth=1)
        add(value, prefix + "-importance-duplicates",
            {"signal": statistic(80, 1), "copy": statistic(80, 1)},
            ["signal", "copy"], ["noise", "constant"])
        value = case("unused", classification,
                     [("signal", "numeric"), ("noise", "numeric")],
                     [[0 if i % 4 == 0 else 1 if i % 4 == 1 else None, (i // 4) % 2] for i in range(80)],
                     [y(i % 4 != 0) for i in range(80)], depth=1)
        add(value, prefix + "-importance-missing", {"signal": statistic(80, 2)},
            ["signal"], ["noise"])
        value = fitting_cases()["ls-mixed-predictors"]
        value["classification"] = classification
        value["options"]["task"] = {"kind": "classification"} if classification else value["options"]["task"]
        value["dataset"]["targets"] = [y(i >= 40) for i in range(80)]
        add(value, prefix + "-importance-mixed",
            {"signal": statistic(80, 3), "noise": statistic(0.8, 3)}, ["signal"], ["group"])
        value = case("unused", classification,
                     [(n, "numeric") for n in ["a", "b", "copy", "noise"]],
                     [[(i // 20) % 2, (i // 40) % 2, (i // 20) % 2, i % 2] for i in range(80)],
                     [y((i // 20) % 2 != (i // 40) % 2) for i in range(80)], depth=1)
        add(value, prefix + "-importance-interaction-copy",
            {n: statistic(80, 3) for n in ["a", "b", "copy"]},
            ["a", "b", "copy"], ["noise"])
    return result
