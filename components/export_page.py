"""Paging for large tables: every row stays loaded, only the rendering is
paged. st.dataframe costs roughly 13s of browser time at 225k rows, so handing
it the full frame freezes the tab."""
import pandas as pd


def page_slice(df: pd.DataFrame, page: int, size: int) -> pd.DataFrame:
    """The `page`-th block of `size` rows, 1-indexed and clamped."""
    pages = max(1, -(-len(df) // size))
    page = min(max(1, page), pages)
    start = (page - 1) * size
    return df.iloc[start:start + size]


def page_count(rows: int, size: int) -> int:
    return max(1, -(-rows // size))
