"""Reading a task out of the benchmark.

Nothing here invents task content. The models to build, their unique keys and
the expected source row counts all come from the benchmark's own files, so a
reviewer comparing our tasks against the published ones finds them identical.

What we do decide is which of those files the agent is shown, because that is
what separates one curriculum stage from the next: with the source tables
already loaded there is nothing for connection details to be used for.
"""
import pytest

from elt_rlvr.task import available_tasks, load_task


def test_targets_come_from_the_benchmarks_data_model_file():
    task = load_task("cs_semester")

    assert set(task.targets) == {"students", "professors", "courses"}


def test_unique_keys_come_from_the_benchmarks_sort_keys():
    task = load_task("cs_semester")

    assert task.keys["courses"] == ("course_id",)
    assert all(model in task.keys for model in task.targets)


def test_expected_source_rows_come_from_the_benchmarks_table_list():
    task = load_task("cs_semester")

    assert task.expected_rows["registration"] == 101
    assert task.expected_rows["course"] == 13


def test_the_agent_is_shown_the_models_to_build_and_the_source_schemas():
    task = load_task("cs_semester")

    assert "data_model.yaml" in task.inputs
    assert "students" in task.inputs["data_model.yaml"]
    assert "schemas/course.csv" in task.inputs
    assert "difficulty of the course" in task.inputs["schemas/course.csv"]


def test_connection_details_are_withheld_while_the_load_is_done_for_the_agent():
    """`config.yaml` describes how to reach the source systems. With the
    tables already in the warehouse there is nothing to reach, and showing it
    would only invite the agent to configure a load it is not being asked for."""
    assert "config.yaml" not in load_task("cs_semester").inputs
    assert "config.yaml" in load_task("cs_semester", include_connections=True).inputs


def test_the_ground_truth_is_not_among_the_files_the_agent_can_read():
    task = load_task("cs_semester")

    assert not any("gt" in name or "ground" in name for name in task.inputs)
    assert all("courses.csv" not in text for text in task.inputs.values())


def test_an_unknown_task_says_so_plainly():
    with pytest.raises(KeyError, match="not_a_task"):
        load_task("not_a_task")


@pytest.mark.needs_benchmark_data
def test_only_tasks_whose_source_data_is_present_are_offered():
    tasks = available_tasks()

    assert "cs_semester" in tasks
    assert "language_corpus" not in tasks  # too large to have been fetched
    assert len(tasks) > 20


@pytest.mark.needs_benchmark_data
def test_every_available_task_loads_and_is_internally_consistent():
    for name in available_tasks():
        task = load_task(name)

        assert task.targets, f"{name} has no targets"
        assert set(task.keys) >= set(task.targets), f"{name} is missing a key"
        assert task.source_dir.is_dir(), f"{name} has no source data"
        for model in task.targets:
            assert (task.ground_truth_dir / f"{model}.csv").is_file(), f"{name}/{model}"
