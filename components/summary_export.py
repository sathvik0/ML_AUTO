"""PowerPoint building blocks shared by the Machine Learning summary deck:
matplotlib figures to PNG, and blank slides with a title, bullets or a table.
"""
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from utils import log

_log = log.get("summary_export")

_W, _H = Inches(13.333), Inches(7.5)
_PURPLE = RGBColor(0x53, 0x4A, 0xB7)
_GREY = RGBColor(0x5F, 0x5E, 0x5A)


def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        # The slide falls back to its table, which is the design — so the log
        # is the only place anyone learns a chart is missing from the deck.
        log.event(_log, "pptx_chart_failed", level=logging.WARNING,
                  exc_info=True, chart=getattr(fn, "__name__", str(fn)))
        return None


# ── pptx layout ──────────────────────────────────────────────────────────

def _slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _title(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = _PURPLE


def _bullets(slide, items, left, top, width, height, size=14):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(size)
        p.space_after = Pt(6)


def _table(slide, df, left, top, width, height, size=11):
    rows, cols = df.shape[0] + 1, df.shape[1]
    tbl = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                 Inches(width), Inches(height)).table
    for j, col in enumerate(df.columns):
        c = tbl.cell(0, j)
        c.text = str(col)
        c.text_frame.paragraphs[0].font.bold = True
        c.text_frame.paragraphs[0].font.size = Pt(size)
    for i in range(df.shape[0]):
        for j in range(cols):
            c = tbl.cell(i + 1, j)
            c.text = str(df.iat[i, j])
            c.text_frame.paragraphs[0].font.size = Pt(size)
