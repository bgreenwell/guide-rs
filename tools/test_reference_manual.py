"""Manual-example batch inputs and their translation, without native GUIDE."""
import unittest
from compare_manual_examples import (FIXTURES, Unsupported, comparable, get, parse_batch,
                                     parse_dsc, translate, unpruned, write_batch)

EXAMPLES = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())
REPORT = " Maximum number of split levels: 12\n Minimum node sample size: 5\n"


# Representative DSC columns: response, numeric, split-only numeric,
# categorical and a flag for x1.
COLUMNS = [("y", "d"), ("x1", "n"), ("x2", "s"), ("g", "c"), ("x1_", "m")]


class ManualExampleTests(unittest.TestCase):
    def test_every_example_is_a_46_3_model_fit(self):
        self.assertEqual(len(EXAMPLES), 13)
        for name in EXAMPLES:
            with self.subTest(example=name):
                header, spec = parse_batch((FIXTURES / name / "input.txt").read_text())
                self.assertIn("46.3", header)
                self.assertEqual(get(spec, "1=model fitting"), "1")
                self.assertEqual(get(spec, "1=one tree"), "1")
                self.assertIsNotNone(get(spec, "name of DSC file"))

    def test_unpruned_batches_drop_cross_validation(self):
        for name in EXAMPLES:
            with self.subTest(example=name):
                header, spec = parse_batch((FIXTURES / name / "input.txt").read_text())
                grown = unpruned(spec)
                pruning = next(v for v, a in grown if "prune by CV" in a)
                self.assertEqual(pruning, "3" if get(spec, "1=classification") == "1" else "2")
                self.assertIsNone(get(grown, "number of cross-validations"))
                # Writing and re-reading preserves every entry.
                self.assertEqual(parse_batch(write_batch(header, grown))[1], grown)

    def test_translations_cover_each_family(self):
        kinds = {}
        for name in EXAMPLES:
            _, spec = parse_batch((FIXTURES / name / "input.txt").read_text())
            spec = comparable(unpruned(spec))
            data, fit = translate(spec, COLUMNS, REPORT, prune=False)
            self.assertEqual(data["target"], "y")
            self.assertEqual((fit["max_depth"], fit["min_samples_leaf"]), (12, 5))
            self.assertEqual(fit["pruning"], {"kind": "none"})
            task = fit["task"].get("loss", fit["task"])["kind"]
            kinds[name] = (task, fit["node_model"]["kind"])
        self.assertEqual(kinds["fish-lda"], ("classification", "lda"))
        self.assertEqual(kinds["ce-polynomial"], ("squared_error", "polynomial"))
        self.assertEqual(kinds["ce-stepwise"], ("squared_error", "stepwise"))
        self.assertEqual(kinds["ce-two-quantiles"], ("dual_quantile", "constant"))
        self.assertEqual(kinds["lungcancer-poisson-offset"], ("poisson", "linear"))

    def test_roles_follow_guide(self):
        _, spec = parse_batch((FIXTURES / "ce-constant" / "input.txt").read_text())
        data, _ = translate(spec, [("y", "d"), ("w", "w"), ("x", "n"), ("x_", "m"), ("o", "z")],
                            REPORT, prune=False)
        self.assertEqual((data["weight"], data["offset"]), ("w", "o"))
        flag = next(f for f in data["features"] if f["name"] == "x_")
        self.assertEqual((flag["role"], flag["flags"]), ("flag", "x"))
        with self.assertRaises(Unsupported):
            translate(spec, [("y", "d"), ("p", "e")], REPORT, prune=False)
        _, spec = parse_batch((FIXTURES / "fish-constant" / "input.txt").read_text())
        data, _ = translate(spec, [("y", "d"), ("x", "n"), ("g", "c")], REPORT, prune=False)
        self.assertEqual([f["role"] for f in data["features"]], ["split", "split"])

    def test_dsc_parsing(self):
        data, missing, start, cols = parse_dsc('data.txt\n"?"\n2\n1 a n\n2, b, d\n')
        self.assertEqual((data, missing, start, cols), ("data.txt", "?", 2, [("a", "n"), ("b", "d")]))


if __name__ == "__main__":
    unittest.main()
