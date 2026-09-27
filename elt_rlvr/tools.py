"""The actions a rollout is made of.

Each method is one thing the agent can ask for, and each returns the text that
goes back into the conversation as its observation. Two rules run through all
of them:

**Nothing raises.** A rollout that hits a broken query, a missing table or a
malformed request is an ordinary rollout, not an error. Letting an exception
escape would abort every trajectory in the GRPO group, discarding the good
ones along with the bad.

**Querying cannot write.** The lineage check re-runs a submission against
altered inputs, which only works if the submission is SQL we kept rather than
a table the agent materialised behind our back. Splitting "look at the
warehouse" from "declare what builds a model" is what makes that possible.

This module deliberately knows nothing about RL. It is bound to the training
framework's tool-calling protocol elsewhere, so the behaviour here can be
tested against a real warehouse without one.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

#: What `run_sql` will run: a query, or a metadata lookup that changes nothing.
#: `SHOW`, `PRAGMA` and `DESCRIBE` are here because rollouts reached for them to
#: inspect a schema and lost a turn each time they were refused.
_READ_ONLY_START = re.compile(r"^(select|with|show|pragma|describe|desc)\b", re.IGNORECASE)

#: Leading comments, which models write above their SQL as a matter of habit.
_LEADING_COMMENTS = re.compile(r"\A(?:\s*(?:--[^\n]*\n|/\*.*?\*/))+", re.DOTALL)

MAX_ROWS = 20
MAX_CHARS = 4000


class EltTools:
    """One instance per rollout, holding that rollout's warehouse and drafts.

    Instantiated inside `make_envs`, so each trajectory gets its own database
    file and its own set of model definitions with nothing shared between them.
    """

    def __init__(
        self,
        warehouse: Any,
        schema: str,
        inputs: dict[str, str],
        targets: list[str],
    ):
        self.warehouse = warehouse
        self.schema = schema
        self._inputs = dict(inputs)
        self._targets = list(targets)
        self._models: dict[str, str] = {}
        self.done = False

    @property
    def submitted_models(self) -> dict[str, str]:
        """The artifact this rollout is graded on: what builds each model.

        Kept as SQL rather than as the resulting tables so the lineage check
        has something to re-run.
        """
        return dict(self._models)

    # -- looking around -------------------------------------------------------

    def read_input(self, name: str) -> str:
        """Read one of the files the task supplied."""
        if name not in self._inputs:
            available = ", ".join(sorted(self._inputs))
            return f"No such input file: {name!r}. Available: {available}"
        return self._inputs[name]

    def peek_source(self, table: str, n: int = 5) -> str:
        """Show a source table's columns and a few of its rows."""
        table = self._unqualify(table)
        try:
            total = self.warehouse.table_size("elt", self.schema, table)
            sample = self.warehouse.query(
                f'SELECT * FROM "{self.schema}"."{table}" LIMIT {int(n)}'
            )
        except Exception as error:
            known = ", ".join(sorted(self.warehouse.list_tables("elt", self.schema)))
            return f"Cannot read {table!r}: {error}\nTables in {self.schema}: {known}"
        return f"{table}: {total} rows\n{self._render(sample, total)}"

    def run_sql(self, query: str) -> str:
        """Run a read-only query and return the rows it produced."""
        refusal = self._refuse_if_not_a_query(query)
        if refusal:
            return refusal
        try:
            rows = self.warehouse.query(query)
        except Exception as error:
            return f"Query error: {error}"
        return self._render(rows, len(rows))

    # -- building -------------------------------------------------------------

    def define_model(self, name: str, sql: str) -> str:
        """Declare the query that builds one target data model.

        Recorded, not run - `run_transform` executes them. Calling this again
        for the same model replaces the previous attempt.
        """
        if name not in self._targets:
            wanted = ", ".join(self._targets)
            return f"{name!r} is not a target of this task. Targets: {wanted}"
        replaced = name in self._models
        self._models[name] = sql
        return f"Recorded the definition of {name}." + (
            " It replaces your previous one." if replaced else ""
        )

    def run_transform(self) -> str:
        """Build every model defined so far, and report how each one went."""
        if not self._models:
            return "No models defined yet. Use define_model first."
        lines = []
        for name, sql in self._models.items():
            try:
                self.warehouse.query(
                    f'CREATE OR REPLACE TABLE "{self.schema}"."{name}" AS {sql}'
                )
                size = self.warehouse.table_size("elt", self.schema, name)
                lines.append(f"{name}: ok, {size} rows")
            except Exception as error:
                # One broken model must not stop the others: the rest still
                # carry signal, and the agent needs every error at once.
                lines.append(f"{name}: failed - {error}")
        return "\n".join(lines)

    def submit(self) -> str:
        """Finish the episode and hand over what has been built."""
        self.done = True
        built = ", ".join(self._models) or "nothing"
        return f"Submitted: {built}."

    # -- helpers --------------------------------------------------------------

    def _unqualify(self, table: str) -> str:
        """Accept `schema.table` as well as `table`, for this task's schema.

        The SQL the agent writes does need the prefix, so it is natural to use
        one here too; refusing it cost rollouts a turn each time. A prefix
        naming a different schema is left alone, so this stays a convenience
        rather than a way to read elsewhere in the warehouse.
        """
        prefix, sep, rest = table.partition(".")
        if sep and prefix.strip('"').lower() == self.schema.lower():
            return rest.strip('"')
        return table


    def _refuse_if_not_a_query(self, sql: str) -> str | None:
        """Reject anything that is not a single statement that only reads.

        Comments are stripped before the check, not skipped over: a write
        wearing a comment is still a write. A prefix check alone would also let
        `SELECT 1; DROP TABLE t` through, so trailing statements are refused.
        """
        stripped = _LEADING_COMMENTS.sub("", sql.strip()).strip().rstrip(";")
        if not _READ_ONLY_START.match(stripped):
            return (
                "run_sql is read-only: it accepts SELECT, WITH, SHOW, PRAGMA "
                "or DESCRIBE. Use define_model to declare how a data model is "
                "built."
            )
        if ";" in stripped:
            return "run_sql is read-only and runs one statement at a time."
        return None

    @staticmethod
    def _render(rows: pd.DataFrame, total: int) -> str:
        """Render a result small enough to leave room for the rest of the task."""
        if rows.empty:
            return "(0 rows)"
        shown = rows.head(MAX_ROWS)
        text = shown.to_string(index=False, max_colwidth=40)
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n... (truncated)"
        if total > len(shown):
            text += f"\n... showing {len(shown)} of {total} rows"
        return text
