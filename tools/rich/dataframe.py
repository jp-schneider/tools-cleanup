# This code is adapted and inspired from https://github.com/khuyentran1401/rich-dataframe/blob/master/rich_dataframe/rich_dataframe.py

from typing import Optional
import pandas as pd
from rich import print
from rich.box import MINIMAL, SIMPLE, SIMPLE_HEAD, SQUARE
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.measure import Measurement
from rich.table import Table
from tools.logger.logging import get_console
COLORS = ["cyan", "magenta", "red", "green", "blue", "purple"]


class RichDataFrame:
    """Create a wrapper around Pandas DataFrame to prettify it using Rich library.

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    row_limit : int, optional
        Number of rows to show, by default 20
    col_limit : int, optional
        Number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show first n rows or last n rows, by default True. If this is set to False, show last n rows.
    first_cols : bool, optional
        Whether to show first n columns or last n columns, by default True. If this is set to False, show last n rows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        row_limit: int = 20,
        col_limit: int = 10,
        first_rows: bool = True,
        first_cols: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        self.df = df.reset_index().rename(columns={"index": "Idx."})
        self.table = Table(show_footer=False)
        self.table_centered = Columns(
            (self.table,), align="center", expand=True
        )
        self.num_colors = len(COLORS)
        self.row_limit = row_limit
        self.first_rows = first_rows
        self.col_limit = col_limit
        self.first_cols = first_cols
        self.console = console if console is not None else get_console()

        if first_cols:
            self.columns = self.df.columns[:col_limit]
        else:
            self.columns = list(self.df.columns[-col_limit:])
            self.columns.insert(0, "index")

        if first_rows:
            self.rows = self.df.values[:row_limit]
        else:
            self.rows = self.df.values[-row_limit:]
        self._add_columns()
        self._add_rows()
        self._add_random_color()
        self._add_style()
        self._set_border_color()
        self._add_caption()

    def _add_columns(self):
        for col in self.columns:
            # Based on dtype add justification
            justify = "left"
            if pd.api.types.is_numeric_dtype(self.df[col]):
                justify = "right"
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                justify = "left"
            self.table.add_column(str(col), justify=justify)

    def _add_rows(self):
        for row in self.rows:
            if self.first_cols:
                row = row[: self.col_limit]
            else:
                row = row[-self.col_limit:]

            row = [str(item) for item in row]
            self.table.add_row(*list(row))

    def _add_random_color(self):
        for i in range(len(self.table.columns)):
            self.table.columns[i].header_style = COLORS[
                i % self.num_colors
            ]

    def _add_style(self):
        for i in range(len(self.table.columns)):
            print(self.table.columns[i].style)
            self.table.columns[i].style = (
                "bold " + COLORS[i % self.num_colors]
            )

    def _adjust_box(self):
        for box in [SIMPLE_HEAD, SIMPLE, MINIMAL, SQUARE]:
            self.table.box = box

    def _dim_row(self):
        self.table.row_styles = ["none", "dim"]

    def _set_border_color(self):
        self.table.border_style = "bright_yellow"

    def _add_caption(self):
        if self.first_rows and len(self.df) > self.row_limit:
            row_text = "first"
        elif len(self.df) > self.row_limit:
            row_text = "last"
        else:
            row_text = "All"
        if self.first_cols and len(self.df.columns) > self.col_limit:
            col_text = "first"
        elif len(self.df.columns) > self.col_limit:
            col_text = "last"
        else:
            col_text = "all"
        self.table.caption = (f"{'Only the' if len(self.df) > self.row_limit or len(self.df.columns) > self.col_limit else ''}" +
                              f"[bold magenta not dim] {row_text} {self.row_limit if len(self.df) > self.row_limit else str(len(self.df))} rows[/bold magenta not dim]" +
                              ("" if len(self.df) <= self.row_limit else "of " + f"[bold green not dim]{len(self.df)} rows[/bold green not dim]") +
                              f" and {'the'} [bold green not dim]{col_text} {self.col_limit if len(self.df.columns) > self.col_limit else str(len(self.df.columns))} columns[/bold green not dim]" +
                              "" if len(self.df.columns) <= self.col_limit else " of " + f"[bold green not dim]{len(self.df.columns)} columns[/bold green not dim]" +
                              " are shown.")
