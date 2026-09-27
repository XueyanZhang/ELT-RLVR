"""What the agent can actually do, and what it cannot.

These are the six actions a rollout is made of. They return text, because text
is what goes back into the conversation, and they never raise, because an
exception here would abort the whole GRPO group rather than one rollout.

The read-only rule on `run_sql` is load-bearing rather than tidiness: the
lineage check re-runs a submission against altered inputs, which is only
possible if the submission is SQL we kept rather than a table the agent built
behind our back.
"""
import pytest

from elt_rlvr.destinations import DuckDBConnector
from elt_rlvr.tools import EltTools

INPUTS = {
    "data_model.yaml": "models:\n  - name: drivers\n    columns: [driver_id, trips]\n",
    "schemas/shipment.csv": "column,description\nship_id,Unique id\ndriver_id,Who drove\n",
}


@pytest.fixture
def tools():
    warehouse = DuckDBConnector({})
    warehouse.query("CREATE SCHEMA shipping")
    warehouse.query(
        "CREATE TABLE shipping.shipment AS SELECT * FROM "
        "(VALUES (1, 20), (2, 20), (3, 25)) t(ship_id, driver_id)"
    )
    kit = EltTools(warehouse, schema="shipping", inputs=INPUTS, targets=["drivers"])
    yield kit
    warehouse.close()


def test_reads_the_task_files_it_was_given(tools):
    assert "drivers" in tools.read_input("data_model.yaml")
    assert "Who drove" in tools.read_input("schemas/shipment.csv")


def test_an_unknown_input_file_lists_what_is_available(tools):
    result = tools.read_input("nope.yaml")

    assert "data_model.yaml" in result  # tells the agent what it can read


def test_peeking_at_a_source_shows_columns_and_rows(tools):
    result = tools.peek_source("shipment")

    assert "ship_id" in result and "driver_id" in result
    assert "3" in result  # the row count is stated


def test_a_query_returns_its_rows(tools):
    result = tools.run_sql("SELECT driver_id, COUNT(*) AS trips FROM shipping.shipment GROUP BY 1 ORDER BY 1")

    assert "20" in result and "25" in result


def test_a_broken_query_returns_the_error_instead_of_raising(tools):
    result = tools.run_sql("SELECT nonexistent FROM shipping.shipment")

    assert "rror" in result  # the database's own message, handed back verbatim


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE shipping.drivers AS SELECT 1 AS driver_id",
        "INSERT INTO shipping.shipment VALUES (9, 9)",
        "DROP TABLE shipping.shipment",
        "UPDATE shipping.shipment SET driver_id = 1",
        "SELECT 1; DROP TABLE shipping.shipment",  # smuggled past a naive prefix check
    ],
)
def test_writing_through_the_query_tool_is_refused(tools, statement):
    result = tools.run_sql(statement)

    assert "read-only" in result.lower()
    assert set(tools.warehouse.list_tables("elt", "shipping")) == {"shipment"}


def test_query_output_is_capped_so_one_call_cannot_fill_the_context(tools):
    tools.warehouse.query(
        "CREATE TABLE shipping.big AS SELECT * FROM range(100000) t(n)"
    )

    result = tools.run_sql("SELECT * FROM shipping.big")

    assert len(result) < 8000
    assert "100000" in result  # but the true size is still reported


def test_defining_a_model_records_it_without_running_it(tools):
    tools.define_model("drivers", "SELECT driver_id FROM shipping.shipment")

    assert tools.submitted_models == {"drivers": "SELECT driver_id FROM shipping.shipment"}
    assert "drivers" not in tools.warehouse.list_tables("elt", "shipping")


def test_redefining_a_model_replaces_the_previous_attempt(tools):
    tools.define_model("drivers", "SELECT 1")
    tools.define_model("drivers", "SELECT 2")

    assert tools.submitted_models["drivers"] == "SELECT 2"


def test_defining_a_model_that_was_not_asked_for_is_refused(tools):
    result = tools.define_model("something_else", "SELECT 1")

    assert "drivers" in result  # names the targets that were asked for
    assert tools.submitted_models == {}


def test_running_the_transform_materialises_every_model(tools):
    tools.define_model(
        "drivers",
        "SELECT driver_id, COUNT(*) AS trips FROM shipping.shipment GROUP BY 1",
    )

    result = tools.run_transform()

    assert "drivers" in tools.warehouse.list_tables("elt", "shipping")
    assert tools.warehouse.table_size("elt", "shipping", "drivers") == 2
    assert "ok" in result.lower()


def test_a_failing_transform_reports_the_database_error(tools):
    tools.define_model("drivers", "SELECT no_such_column FROM shipping.shipment")

    result = tools.run_transform()

    assert "drivers" in result and "no_such_column" in result


def test_one_failing_model_does_not_stop_the_others(tools):
    tools.define_model("drivers", "SELECT driver_id FROM shipping.shipment")
    tools._models["extra"] = "SELECT broken"  # bypass the target check

    tools.run_transform()

    assert "drivers" in tools.warehouse.list_tables("elt", "shipping")


def test_running_the_transform_before_defining_anything_says_so(tools):
    assert "no" in tools.run_transform().lower()


def test_submitting_ends_the_episode(tools):
    assert tools.done is False

    tools.submit()

    assert tools.done is True


@pytest.mark.parametrize(
    "statement",
    [
        "-- find the most popular professor\nSELECT prof_id FROM shipping.shipment",
        "/* a block comment */ SELECT ship_id FROM shipping.shipment",
        "\n\n-- two\n-- comment lines\nWITH x AS (SELECT 1) SELECT * FROM x",
    ],
)
def test_a_query_introduced_by_a_comment_is_still_a_query(tools, statement):
    """Models write a comment above their SQL, and the first runs showed the
    read-only check rejecting perfectly ordinary queries because of it - at the
    cost of a turn each time."""
    result = tools.run_sql(statement)

    assert "read-only" not in result.lower()


@pytest.mark.parametrize(
    "statement",
    ["SHOW TABLES", "PRAGMA table_info('shipping.shipment')", "DESCRIBE shipping.shipment"],
)
def test_read_only_introspection_is_allowed(tools, statement):
    """`SHOW` and `PRAGMA` read metadata and change nothing. Refusing them sent
    rollouts looking for another way to see a schema."""
    result = tools.run_sql(statement)

    assert "read-only" not in result.lower()


def test_a_write_hidden_behind_a_comment_is_still_refused(tools):
    """The comment-stripping must not become a way around the check."""
    result = tools.run_sql("-- just looking\nDROP TABLE shipping.shipment")

    assert "read-only" in result.lower()
    assert set(tools.warehouse.list_tables("elt", "shipping")) == {"shipment"}


@pytest.mark.parametrize("name", ["shipment", "shipping.shipment", "SHIPPING.Shipment"])
def test_a_source_can_be_named_with_or_without_its_schema(tools, name):
    """The briefing tells the agent that every reference to a source table
    needs its schema prefix, which is true of the SQL it writes. Rollouts
    reasonably applied the same rule here and lost turns to
    `world."world.countries"` - our own instruction and our own tool disagreed
    about what a table is called."""
    result = tools.peek_source(name)

    assert "3 rows" in result
    assert "does not exist" not in result


def test_a_table_from_another_schema_is_still_refused(tools):
    """Tolerating the task's own prefix must not become a way to read
    elsewhere in the warehouse."""
    result = tools.peek_source("other_schema.secrets")

    assert "Cannot read" in result
