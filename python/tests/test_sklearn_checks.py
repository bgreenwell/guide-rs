from sklearn.utils.estimator_checks import parametrize_with_checks

from guide_rs import GuideClassifier, GuideForestClassifier, GuideForestRegressor, GuideRegressor


def expected_failures(estimator):
    # GUIDE's selection tables count rows rather than summing weights, and
    # classification treats every positive weight as 1, so a weight of 2 is
    # not equivalent to a repeated row.
    return {
        "check_sample_weight_equivalence_on_dense_data": "weights are not row counts in GUIDE",
    }


@parametrize_with_checks(
    [
        GuideRegressor(),
        GuideClassifier(),
        GuideForestRegressor(n_estimators=5),
        GuideForestClassifier(n_estimators=5),
    ],
    expected_failed_checks=expected_failures,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
