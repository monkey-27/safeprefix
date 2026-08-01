from safeprefix.prefix_validity_v1.reporting import compact_localization_table


def test_compact_localization_table_keeps_hidden_and_position_separate() -> None:
    checkpoint = {"roc_auc": 0.7, "average_precision": 0.8}
    boundary = {
        "exact_last_valid_checkpoint_accuracy": 0.4,
        "within_one_checkpoint_accuracy": 0.6,
        "mean_absolute_checkpoint_error": 1.2,
        "late_boundary_rate": 0.04,
        "early_boundary_rate": 0.5,
        "mean_retained_valid_prefix_fraction": 0.7,
        "non_root_predicted_boundary_coverage": 0.8,
    }
    metrics = {"m": {
        "linear_probe": {"overall": {"checkpoint": checkpoint, "boundary": boundary}},
        "position_only": {"overall": {"checkpoint": checkpoint, "boundary": boundary}},
    }}
    table = compact_localization_table(metrics)
    assert set(table["probe"]) == {"linear_probe", "position_only"}
    assert table["late_boundary_rate"].max() == 0.04
