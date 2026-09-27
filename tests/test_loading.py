"""Putting a task's source tables into a warehouse.

This is the load half of ELT, done for the agent rather than by it. It gives
the transform-only curriculum a warehouse to start from, and it builds the
template file that every rollout of a task is reset to - so it runs once per
task, not once per episode.

Row counts are checked against the benchmark's own expectations as the tables
land: a silently short load would show up later as a low reward and be blamed
on the policy.
"""
import pytest

from elt_rlvr.destinations import DuckDBConnector
from elt_rlvr.loading import load_sources


@pytest.fixture
def source_dir(tmp_path):
    task = tmp_path / "shipping"
    task.mkdir()
    (task / "driver.csv").write_text("driver_id,first_name\n20,Sue\n25,Leszek\n")
    (task / "city.csv").write_text("city_id,name\n1,Lviv\n")
    return task


def test_every_csv_becomes_a_table_in_the_task_schema(source_dir):
    warehouse = DuckDBConnector({})

    load_sources(warehouse, "shipping", source_dir)

    assert set(warehouse.list_tables("elt", "shipping")) == {"driver", "city"}
    assert warehouse.table_size("elt", "shipping", "driver") == 2


def test_loaded_rows_are_verified_against_the_benchmark_row_counts(source_dir):
    warehouse = DuckDBConnector({})

    report = load_sources(
        warehouse, "shipping", source_dir, expected={"driver": 2, "city": 1}
    )

    assert report.ok is True
    assert report.mismatches == {}


def test_a_short_load_is_reported_rather_than_passing_quietly(source_dir):
    warehouse = DuckDBConnector({})

    report = load_sources(
        warehouse, "shipping", source_dir, expected={"driver": 99, "city": 1}
    )

    assert report.ok is False
    assert report.mismatches == {"driver": (2, 99)}


def test_a_table_the_benchmark_expects_but_that_is_absent_is_reported(source_dir):
    warehouse = DuckDBConnector({})

    report = load_sources(
        warehouse, "shipping", source_dir, expected={"driver": 2, "city": 1, "truck": 7}
    )

    assert report.ok is False
    assert "truck" in report.missing


def test_loading_twice_leaves_one_copy(source_dir):
    warehouse = DuckDBConnector({})

    load_sources(warehouse, "shipping", source_dir)
    load_sources(warehouse, "shipping", source_dir)

    assert warehouse.table_size("elt", "shipping", "driver") == 2


def test_columns_arrive_as_text_so_comparison_is_not_at_the_mercy_of_inference(source_dir):
    """The grader reads both sides as strings; loading with inferred types
    would let the warehouse decide, for instance, whether `007` is a number."""
    warehouse = DuckDBConnector({})

    load_sources(warehouse, "shipping", source_dir)

    types = warehouse.query(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = 'shipping' AND table_name = 'driver'"
    )
    assert set(types["data_type"]) == {"VARCHAR"}


def test_row_counts_are_matched_regardless_of_identifier_case(tmp_path):
    """A file on disk and the benchmark's table list need not agree on case -
    `seasonstatus.csv` is what ships for a table the benchmark calls
    `SeasonStatus`. The warehouse folds case, so the check must too, or a
    correctly loaded table is reported as missing."""
    task = tmp_path / "ice_hockey_draft"
    task.mkdir()
    (task / "seasonstatus.csv").write_text("player_id,team\n1,Kings\n")
    warehouse = DuckDBConnector({})

    report = load_sources(warehouse, "hockey", task, expected={"SeasonStatus": 1})

    assert report.ok is True
    assert report.missing == ()


def test_only_the_tables_the_benchmark_declares_are_loaded(tmp_path):
    """The archive holds files a task does not declare - `trains` ships a
    duplicate of one source table under the task's own name, which collides
    with the name of the data model the task asks for. Loading it would put a
    table where the submission is supposed to go, and an agent that did nothing
    would be graded on it."""
    task = tmp_path / "trains"
    task.mkdir()
    (task / "cars.csv").write_text("train_id,len\n1,long\n")
    (task / "train.csv").write_text("id,direction\n1,east\n")
    (task / "trains.csv").write_text("id,direction\n1,east\n")  # undeclared
    warehouse = DuckDBConnector({})

    load_sources(warehouse, "trains", task, expected={"cars": 1, "train": 1})

    assert set(warehouse.list_tables("elt", "trains")) == {"cars", "train"}


def test_without_an_expected_table_list_every_csv_is_loaded(tmp_path):
    """Callers that have no list - the tests above, or exploratory use - still
    get everything."""
    task = tmp_path / "anything"
    task.mkdir()
    (task / "a.csv").write_text("x\n1\n")
    (task / "b.csv").write_text("y\n2\n")
    warehouse = DuckDBConnector({})

    load_sources(warehouse, "anything", task)

    assert set(warehouse.list_tables("elt", "anything")) == {"a", "b"}
