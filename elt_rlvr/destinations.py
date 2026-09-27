"""Warehouse destinations, including a local one.

The benchmark already separates warehouse-specific behaviour - connections,
identifier casing, information-schema queries, table fetches - from its
evaluation logic, behind a single `BaseConnector`. That seam is what the task
calls for: support one warehouse, but leave room for others without rewriting
the environment.

We add DuckDB behind the same interface and leave the benchmark's own
connectors untouched. A rollout then runs against a local file, with the same
environment and the same grader that a credentialed run uses.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from elt_rlvr._official import official_stage2


def _base_connector():
    """The benchmark's connector base class, from the pinned submodule."""
    official_stage2()  # puts the evaluation package on sys.path
    import db_connectors

    return db_connectors


class DuckDBConnector(_base_connector().BaseConnector):
    """A warehouse in a single local file.

    One file per rollout, so concurrent episodes cannot see each other and a
    reset is a file copy rather than a schema rebuild (§5).

    Layout: task schemas sit directly in the file, holding both the loaded
    source tables and the data models built from them - the same arrangement
    Snowflake gets from `<task>.AIRBYTE_SCHEMA.<table>`, minus a level. No
    source table shares a name with a data model anywhere in the benchmark, so
    one namespace is enough.

    Config: ``{"path": "/some/file.duckdb"}``, or ``{}`` for an in-memory
    warehouse, which is what the tests and the credential-free path use.
    """

    def _connect(self):
        import duckdb

        return duckdb.connect(self.config.get("path") or ":memory:")

    def _quote(self, ident: str) -> str:
        return f'"{ident}"'

    def _norm(self, ident: str) -> str:
        """DuckDB resolves unquoted identifiers case-insensitively."""
        return ident.lower()

    def _fqn(self, database: str, schema: str, table: str) -> str:
        """`database` is the file itself, so a task schema needs no prefix."""
        return f"{self._quote(schema)}.{self._quote(table)}"

    def _schemata_sql(self, database: str) -> str:
        return "SELECT schema_name FROM information_schema.schemata"

    def _tables_sql(self, database: str, schema: str) -> str:
        return (
            "SELECT table_name FROM information_schema.tables "
            f"WHERE lower(table_schema) = '{schema.lower()}' "
            "AND table_type = 'BASE TABLE'"
        )

    def query(self, sql: str) -> pd.DataFrame:
        """DuckDB hands back a dataframe directly, with no cursor dance."""
        return self.conn().execute(sql).fetchdf()

    # `prepare_evaluation_sql` is inherited unchanged: the benchmark's checks
    # are written as `schema.table`, which is already this layout.


def get_connector(db_type: str, config: dict[str, Any]):
    """Resolve a destination by name.

    Mirrors the benchmark's own factory and falls through to it, so the three
    warehouses it supports keep working exactly as before.
    """
    if db_type == "duckdb":
        return DuckDBConnector(config)
    return _base_connector().get_connector(db_type, config)
