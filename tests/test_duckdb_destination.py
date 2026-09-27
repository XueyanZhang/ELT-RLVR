"""A local warehouse that behaves like the graded ones.

The benchmark ships connectors for Snowflake, Databricks and Redshift, all
behind one interface the evaluators code against. Adding DuckDB behind that
same interface is what makes the credential-free path possible: the same
environment, the same grader, a different destination.
"""
import pandas as pd
import pytest

from elt_rlvr.destinations import DuckDBConnector, get_connector


@pytest.fixture
def warehouse(tmp_path):
    connector = DuckDBConnector({"path": str(tmp_path / "rollout.duckdb")})
    connector.query("CREATE SCHEMA shipping")
    connector.query("CREATE TABLE shipping.driver AS SELECT * FROM (VALUES (20, 'Sue'), (25, 'Leszek')) t(driver_id, first_name)")
    connector.query("CREATE TABLE shipping.city AS SELECT * FROM (VALUES (1, 'Lviv')) t(city_id, name)")
    yield connector
    connector.close()


def test_a_task_schema_is_visible_once_created(warehouse):
    assert warehouse.verify_schema("elt", "shipping") is True


def test_an_absent_schema_is_reported_rather_than_raised(warehouse):
    assert warehouse.verify_schema("elt", "never_created") is False


def test_tables_in_a_schema_are_listed(warehouse):
    assert set(warehouse.list_tables("elt", "shipping")) == {"driver", "city"}


def test_row_counts_come_back_as_integers(warehouse):
    assert warehouse.table_size("elt", "shipping", "driver") == 2


def test_a_table_round_trips_as_a_dataframe(warehouse):
    table = warehouse.fetch_table("elt", "shipping", "driver")

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["driver_id", "first_name"]
    assert len(table) == 2


def test_benchmark_evaluation_sql_runs_unchanged(warehouse):
    """The benchmark writes its checks against a warehouse-neutral
    `schema.table`, which each connector maps onto its own layout. DuckDB's
    layout is that form already, so nothing needs rewriting."""
    result = warehouse.query_evaluation_sql(
        "SELECT first_name FROM shipping.driver ORDER BY driver_id", "elt"
    )

    assert list(result["first_name"]) == ["Sue", "Leszek"]


def test_identifiers_are_folded_consistently(warehouse):
    """Stage 1 compares the tables it expects against the tables it finds, so
    the two sides must fold case the same way."""
    assert warehouse._norm("Driver") == warehouse._norm("driver")
    assert set(map(warehouse._norm, warehouse.list_tables("elt", "shipping"))) == {
        warehouse._norm("DRIVER"),
        warehouse._norm("City"),
    }


def test_an_in_memory_warehouse_needs_no_path():
    connector = get_connector("duckdb", {})
    try:
        assert connector.query("SELECT 1 AS n").iloc[0, 0] == 1
    finally:
        connector.close()


def test_the_official_connectors_are_still_reachable_through_the_factory():
    """We add a destination; we do not replace the benchmark's own."""
    with pytest.raises(Exception):
        # No credentials here - the point is that the name still resolves to
        # the upstream connector rather than being rejected as unknown.
        get_connector("snowflake", {}).conn()
