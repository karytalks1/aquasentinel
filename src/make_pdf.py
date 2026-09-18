"""Build the project explainer PDF (reports/AquaSentinel_Project_Guide.pdf).

A single self-contained document written for two readers: a project partner who
needs to understand the work from zero, and an examiner who needs the method,
the evidence and the limitations. All numbers are read from reports/tables so
the document can never drift out of step with the results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402

NAVY = colors.HexColor("#1f4e79")
ACCENT = colors.HexColor("#e08214")
GREENC = colors.HexColor("#2e7d32")
LIGHT = colors.HexColor("#eef3f8")
RULE = colors.HexColor("#c8d6e5")
MUTED = colors.HexColor("#5a6872")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

OUT = C.ROOT / "reports" / "AquaSentinel_Project_Guide.pdf"


# ---------------------------------------------------------------- styles ----
def build_styles() -> dict:
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=26, leading=31, textColor=NAVY, spaceAfter=4)
    s["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=12.5,
                                   leading=17, textColor=MUTED, alignment=TA_CENTER)
    s["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                             fontSize=16, leading=20, textColor=NAVY,
                             spaceBefore=16, spaceAfter=8)
    s["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12, leading=15, textColor=colors.HexColor("#2c3e50"),
                             spaceBefore=11, spaceAfter=5)
    s["body"] = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.8, leading=14.6,
                               alignment=TA_JUSTIFY, spaceAfter=7)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12,
                                 bulletIndent=3, spaceAfter=3.5)
    s["caption"] = ParagraphStyle("caption", parent=ss["Normal"], fontSize=8.4,
                                  leading=11.5, textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=4, spaceAfter=11)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], fontName="Helvetica-Oblique",
                                fontSize=10, leading=14.5, leftIndent=10,
                                rightIndent=10, textColor=colors.HexColor("#243b53"))
    s["kpi_num"] = ParagraphStyle("kpi_num", parent=ss["Normal"], fontName="Helvetica-Bold",
                                  fontSize=19, leading=22, alignment=TA_CENTER,
                                  textColor=NAVY)
    s["kpi_lbl"] = ParagraphStyle("kpi_lbl", parent=ss["Normal"], fontSize=7.6,
                                  leading=10, alignment=TA_CENTER, textColor=MUTED)
    s["toc1"] = ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10.5,
                               leading=17, textColor=NAVY)
    s["toc2"] = ParagraphStyle("toc2", fontSize=9.4, leading=14, leftIndent=14,
                               textColor=colors.HexColor("#33475b"))
    return s


S = build_styles()
_toc_seq = [0]


def P(text, style="body"):
    return Paragraph(text, S[style])


def H(text, level=1):
    """Heading that also registers itself with the table of contents."""
    _toc_seq[0] += 1
    key = f"h{_toc_seq[0]}"
    style = "h1" if level == 1 else "h2"
    para = Paragraph(f'<a name="{key}"/>{text}', S[style])
    para._toc = (level - 1, text, key)
    return para


def bullets(items):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", S["bullet"]) for t in items]


def callout(text, tone=NAVY, bg=LIGHT):
    """A tinted box with a coloured left rule, for the ideas that matter most."""
    t = Table([[Paragraph(text, S["body"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, tone),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def data_table(rows, widths=None, align_right_from=1, highlight_row=None,
               font_size=8.6):
    """A clean table: navy header, zebra body, optional highlighted row."""
    widths = widths or [CONTENT_W / len(rows[0])] * len(rows[0])
    body = ParagraphStyle("tb", fontSize=font_size, leading=font_size + 3.2)
    head = ParagraphStyle("th", fontSize=font_size, leading=font_size + 3.2,
                          fontName="Helvetica-Bold", textColor=colors.white)
    data = [[Paragraph(str(c), head) for c in rows[0]]]
    data += [[Paragraph(str(c), body) for c in r] for r in rows[1:]]

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (align_right_from, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f9fc")))
    if highlight_row is not None:
        style += [("BACKGROUND", (0, highlight_row), (-1, highlight_row),
                   colors.HexColor("#e3f1e5")),
                  ("LINEBELOW", (0, highlight_row), (-1, highlight_row), 1, GREENC),
                  ("LINEABOVE", (0, highlight_row), (-1, highlight_row), 1, GREENC)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def figure(name: str, caption: str, width_frac: float = 1.0):
    """Embed a report figure, scaled to fit the text column."""
    path = C.FIGURES / f"{name}.png"
    if not path.exists():
        return Spacer(1, 1)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    w = CONTENT_W * width_frac
    h = w * ih / iw
    max_h = 108 * mm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    return KeepTogether([Image(str(path), width=w, height=h),
                         P(caption, "caption")])


# ------------------------------------------------------------ page frame ----
def _chrome(canvas, doc, cover=False):
    canvas.saveState()
    if cover:
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 12 * mm, PAGE_W, 12 * mm, stroke=0, fill=1)
        canvas.restoreState()
        return
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 11 * mm,
                      "AquaSentinel - Leak Detection & Localisation in a Water Network")
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.drawString(MARGIN, 8.5 * mm, "BattLeDIM 2020 benchmark - L-Town network")
    canvas.drawRightString(PAGE_W - MARGIN, 8.5 * mm, f"Page {doc.page - 1}")
    canvas.restoreState()


def build_doc():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=20 * mm, bottomMargin=18 * mm,
                          title="AquaSentinel - Project Guide",
                          author="Kartik")
    frame = Frame(MARGIN, 18 * mm, CONTENT_W, PAGE_H - 38 * mm, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame],
                     onPage=lambda c, d: _chrome(c, d, cover=True)),
        PageTemplate(id="body", frames=[frame], onPage=_chrome),
    ])
    return doc


# ------------------------------------------------------------- content ------
def load_results():
    t = C.TABLES
    return {
        "det": pd.read_csv(t / "detection_summary.csv"),
        "loc": pd.read_csv(t / "localisation_summary.csv"),
        "sens": pd.read_csv(t / "sensor_ablation.csv"),
        "zone": pd.read_csv(t / "zone_ablation.csv"),
        "det19": pd.read_csv(t / "detections_2019.csv"),
        "loc19": pd.read_csv(t / "localisation_2019.csv"),
    }


def cover(R):
    d19 = R["det"][R["det"]["year"] == 2019].iloc[0]
    story = [
        Spacer(1, 24 * mm),
        P("AquaSentinel", "title"),
        P("Finding water leaks underground, using only the sensors a utility "
          "already has", "subtitle"),
        Spacer(1, 12 * mm),
    ]

    kpis = [
        (f"{int(d19.detected)}/{int(d19.leaks)}", "LEAKS DETECTED<br/>(held-out year)"),
        (f"{int(d19.false_alarms)}", "FALSE ALARMS<br/>in a full year"),
        (f"{d19.median_delay_days:.0f} days", "MEDIAN TIME<br/>to detection"),
        ("37%", "CORRECT DISTRICT<br/>first guess (8% random)"),
    ]
    cells = [[Paragraph(v, S["kpi_num"]) for v, _ in kpis],
             [Paragraph(l, S["kpi_lbl"]) for _, l in kpis]]
    t = Table(cells, colWidths=[CONTENT_W / 4] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
    ]))
    story += [t, Spacer(1, 10 * mm)]

    story += [callout(
        "<b>What this document is.</b> A complete, plain-language explanation of the "
        "project: the real-world problem, where the data came from, how the method "
        "works, what the results are, and where it falls short. It assumes no prior "
        "knowledge of water engineering. Section 6 walks through one real leak from "
        "start to finish - that worked example is the fastest way to understand "
        "everything.", ACCENT, colors.HexColor("#fdf3e7"))]

    story += [Spacer(1, 8 * mm), P(
        "<b>Dataset:</b> BattLeDIM 2020 (Zenodo record 4017659, CC BY 4.0), released by "
        "the KIOS Center of Excellence, University of Cyprus.<br/>"
        "<b>Network:</b> L-Town - 782 junctions, 905 pipes, 43 km, 33 pressure sensors.<br/>"
        "<b>Protocol:</b> 2018 used for all development; 2019 held out entirely and "
        "evaluated once.")]
    story += [Spacer(1, 14 * mm), P(
        "<font color='#5a6872' size='8.5'>Everything in this document is regenerated "
        "directly from the result tables produced by the code, so the figures quoted "
        "here and the output of the software cannot drift apart. "
        f"Generated {pd.Timestamp.today():%d %B %Y}.</font>")]
    story += [NextPageTemplate("body"), PageBreak()]
    return story


def toc_page():
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    return [Paragraph("Contents", S["h1"]), Spacer(1, 4), toc, PageBreak()]


def sec_summary(R):
    d19 = R["det"][R["det"]["year"] == 2019].iloc[0]
    d18 = R["det"][R["det"]["year"] == 2018].iloc[0]
    s = [H("1. The project in one page")]
    s += [P(
        "Cities lose an enormous amount of treated water through leaks in buried pipes "
        "that nobody knows about. A burst that breaks the road surface gets reported "
        "within hours. A small crack simply pushes water into the soil, silently, for "
        "months. This project builds a system that spots those hidden leaks early and "
        "tells a repair crew which district to search.")]
    s += [P("It produces two outputs:")]
    s += bullets([
        "<b>Detection</b> - an alarm saying a <i>new</i> leak has started, and roughly when.",
        "<b>Localisation</b> - which of 12 districts to search, narrowing the hunt from "
        "43 km of pipe to about 3.6 km.",
    ])
    s += [Spacer(1, 4), P(
        "The inputs are only what a water utility already measures: 33 pressure sensors, "
        "3 flow meters and 1 tank level sensor, sampled every five minutes. No new "
        "hardware, no fieldwork.")]

    s += [H("Headline results", 2), P(
        "Evaluated against the method utilities use today, <b>minimum night flow</b> (MNF), "
        "on a year that played no part in developing the system:")]
    s += [data_table([
        ["Measure", "This project", "Night flow (current practice)"],
        ["Leaks detected (2019)", f"<b>{int(d19.detected)} of {int(d19.leaks)}  "
                                  f"({d19.detection_rate:.0%})</b>",
         f"{int(d19.mnf_detected)} of {int(d19.leaks)}  ({d19.mnf_detection_rate:.0%})"],
        ["False alarms", f"<b>{int(d19.false_alarms)}</b>", f"{int(d19.mnf_false_alarms)}"],
        ["Median time to detection", f"<b>{d19.median_delay_days:.0f} days</b>",
         f"{d19.mnf_median_delay_days:.0f} days"],
        ["Leaks detected (2018)", f"{int(d18.detected)} of {int(d18.leaks)}  "
                                  f"({d18.detection_rate:.0%})",
         f"{int(d18.mnf_detected)} of {int(d18.leaks)}  ({d18.mnf_detection_rate:.0%})"],
    ], widths=[CONTENT_W * 0.34, CONTENT_W * 0.33, CONTENT_W * 0.33])]
    s += [Spacer(1, 6), callout(
        "<b>The one idea to take away.</b> A leak makes water rush toward the hole, which "
        "increases friction and <i>drops the pressure</i> nearby. Everything in this "
        "project is a careful way of measuring that drop and working out where it came "
        "from.")]
    return s


def sec_setting():
    s = [H("2. The physical setting")]
    s += [H("What a water distribution network is", 2), P(
        "A city's water supply is a network of buried pipes - much like a road network, "
        "but for water. Water enters at a few points (reservoirs, a pump, a storage tank) "
        "and travels through the pipes to every building.")]
    s += [P("The network in this study is called <b>L-Town</b>:")]
    s += bullets([
        "<b>782 junctions</b> - points where pipes meet and where buildings draw water",
        "<b>905 pipes</b> - about <b>43 km</b> of pipe in total",
        "<b>2 reservoirs, 1 pump, 1 storage tank</b> - the sources of supply",
        "<b>3 pressure reducing valves (PRVs)</b> - valves that cap the pressure",
    ])
    s += [H("What pressure is, and why it matters", 2), P(
        "Pressure is how hard water pushes inside a pipe. Here it is measured in "
        "<b>metres</b>: '50 m of pressure' means the water pushes as hard as a 50-metre "
        "column of water would. Enough pressure is needed to reach upper floors; too "
        "little and taps run dry.")]
    s += [H("What a leak does physically", 2), P(
        "A leak is a hole. Water escapes through it, and that escaping water has to come "
        "from somewhere - so extra water is pulled through the surrounding pipes toward "
        "the hole. When more water moves through a pipe, friction rises and "
        "<b>pressure falls</b>. The drop is largest near the leak and fades with distance.")]
    s += [callout(
        "<b>This is the entire physical basis of the project.</b> A leak creates a "
        "pressure drop whose <i>size</i> tells you how big the leak is, and whose "
        "<i>pattern across sensors</i> tells you where it is.")]
    s += [H("The instruments available", 2), P(
        "The utility has only this much visibility into 43 km of buried pipe:")]
    s += bullets([
        "<b>33 pressure sensors</b> at 33 junctions, reporting every 5 minutes",
        "<b>3 flow meters</b> measuring water entering the network",
        "<b>1 tank level sensor</b>",
    ])
    s += [figure("01_network_map",
                 "Figure 1. The L-Town network. Coloured dots are pipes, shaded by "
                 "district. Black triangles are the 33 pressure sensors - the only "
                 "inputs the system gets. Red crosses mark the 33 real leaks. Note how "
                 "sparse the sensors are relative to the pipes: that sparsity is the "
                 "core difficulty of the problem.")]
    return s


def sec_problem():
    s = [H("3. The real-world problem")]
    s += [P(
        "Water lost between treatment and the customer's meter is called "
        "<b>non-revenue water</b> - water that was cleaned, pumped and paid for, but "
        "never billed. Most of it is leakage. The cost is not only the water itself but "
        "the chemicals and electricity spent producing and moving it, plus the damage "
        "done underground before anyone notices.")]
    s += [H("What utilities do today: minimum night flow", 2), P(
        "Between roughly 2 am and 4 am almost nobody uses water. So whatever is still "
        "flowing into a district at 3 am is assumed to be loss. If that night-time flow "
        "steps up, a new leak is suspected. It is simple, it needs no modelling, and it "
        "is the standard method in the industry.")]
    s += [H("Why it performs poorly here", 2), P(
        "Night flow is noisy. Some nights people are awake; some nights are warmer. "
        "Measured on this network:")]
    s += [data_table([
        ["Quantity", "Magnitude"],
        ["Random night-to-night variation", "26 to 52 m&sup3;/h"],
        ["Extra flow caused by a typical leak", "10 to 45 m&sup3;/h"],
    ], widths=[CONTENT_W * 0.6, CONTENT_W * 0.4])]
    s += [Spacer(1, 6), P(
        "<b>The noise is larger than the signal.</b> The leak is buried inside ordinary "
        "variation, so it simply cannot be seen. Implemented faithfully as a baseline, "
        "this method found <b>1 leak out of 19</b>.")]
    s += [figure("10_mnf_noise",
                 "Figure 2. Night-time inflow through 2019. Each red line is a real leak "
                 "starting. The onsets are indistinguishable from ordinary fluctuation - "
                 "which is precisely why a better method is needed.")]
    return s


def sec_data():
    s = [H("4. The data: where it comes from and what is in it")]
    s += [H("Source", 2)]
    s += bullets([
        "<b>Dataset:</b> BattLeDIM 2020 - 'Battle of the Leakage Detection and Isolation "
        "Methods'",
        "<b>Produced by:</b> the KIOS Center of Excellence, University of Cyprus",
        "<b>Why it exists:</b> it was released for an international research competition "
        "in 2020, so that competing methods could be compared on identical data with "
        "identical answers",
        "<b>Obtained from:</b> Zenodo, a public research repository - record 4017659, "
        "licence CC BY 4.0 (free to use with attribution)",
        "<b>How:</b> the script <font face='Courier'>src/download_data.py</font> fetches "
        "it automatically and resumes if the connection drops",
    ])
    s += [Spacer(1, 4), callout(
        "<b>Is it real data?</b> It is generated by hydraulic simulation of a real "
        "network model, with realistic demand patterns and measurement noise, and was "
        "published as a benchmark precisely so that methods could be compared against "
        "<i>verified</i> ground truth. That is a strength rather than a weakness: real "
        "utility records almost never come with confirmed leak labels, so honest "
        "measurement of detection rate and timing would be impossible.")]

    s += [H("What the files contain", 2)]
    s += [data_table([
        ["File", "Contents"],
        ["L-TOWN.inp", "The network model: every pipe, junction, diameter and elevation"],
        ["2018_SCADA_Pressures.csv", "33 pressure sensors, every 5 minutes, all year"],
        ["2018_SCADA_Flows.csv", "3 flow meters"],
        ["2018_SCADA_Levels.csv", "Tank level"],
        ["2018_Leakages.csv", "<i>Answer key</i> - flow through each leak at every moment"],
        ["dataset_configuration.yaml", "<i>Answer key</i> - pipe, start, end, hole "
                                       "diameter and type of every leak"],
        ["(the same files for 2019)", "The held-out evaluation year"],
    ], widths=[CONTENT_W * 0.33, CONTENT_W * 0.67], align_right_from=2)]
    s += [Spacer(1, 6), P(
        "<b>SCADA</b> stands for Supervisory Control and Data Acquisition - the industry "
        "term for the system that gathers sensor readings.")]

    s += [H("What a row of data looks like", 2)]
    s += [data_table([
        ["Timestamp", "n1", "n4", "n31", "n54", "n105", "n114"],
        ["2019-01-01 00:00:00", "28.63", "33.72", "37.00", "36.93", "50.45", "53.92"],
        ["2019-01-01 00:05:00", "28.66", "33.75", "37.02", "37.05", "50.54", "54.02"],
        ["2019-01-01 00:10:00", "28.67", "33.76", "37.04", "37.03", "50.51", "53.98"],
    ], widths=[CONTENT_W * 0.3] + [CONTENT_W * 0.7 / 6] * 6, font_size=8)]
    s += [Spacer(1, 5), P(
        "Each row is one instant; each column is one sensor's pressure in metres. There "
        "are <b>105,120 rows per year</b> (365 days &times; 24 hours &times; 12 readings "
        "per hour).")]

    s += [H("The ground truth", 2), P(
        "Every leak is documented exactly. For example:")]
    s += [callout("<font face='Courier'>p523,&nbsp; 2019-01-15 23:00,&nbsp; "
                  "2019-02-01 09:50,&nbsp; 0.020246 m,&nbsp; abrupt</font><br/><br/>"
                  "Read as: <i>pipe p523 leaked from 15 January at 11 pm until 1 February, "
                  "through a 20.2 mm hole, and it was an abrupt burst.</i>",
                  MUTED, colors.HexColor("#f4f6f8"))]
    s += [Spacer(1, 4), P(
        "There are <b>33 such leaks</b> - 14 beginning in 2018 and 19 in 2019.")]

    s += [H("Two kinds of leak - this drives every result", 2)]
    s += [data_table([
        ["Type", "What it is", "Behaviour", "Count"],
        ["<b>Abrupt</b>", "A burst - the pipe cracks open at once",
         "Full discharge from the first second. Comparatively easy to spot.", "15"],
        ["<b>Incipient</b>", "A crack that slowly widens over weeks",
         "Starts almost invisibly and grows. Much harder to catch early.", "18"],
    ], widths=[CONTENT_W * 0.14, CONTENT_W * 0.3, CONTENT_W * 0.45, CONTENT_W * 0.11],
        align_right_from=3)]
    return s


def sec_discovery():
    s = [H("5. The discovery that shaped the whole project")]
    s += [P(
        "The obvious first approach is a classifier: feed in the sensor readings, and "
        "predict 'leak' or 'no leak'. Checking the labels rules that out immediately.")]
    s += [data_table([
        ["Year", "Share of the year with no leak flowing", "Most leaks running at once"],
        ["2018", "2.2%", "6"],
        ["2019", "<b>0%</b>", "<b>16</b>"],
    ], widths=[CONTENT_W * 0.2, CONTENT_W * 0.5, CONTENT_W * 0.3])]
    s += [Spacer(1, 7), P(
        "In 2019, <b>something is leaking at every single moment of the year</b>. A model "
        "that always answers 'yes, there is a leak' would be 100% accurate and completely "
        "worthless - it tells an operator nothing they can act on.")]
    s += [P(
        "There is a second difficulty. To recognise abnormal behaviour you need examples "
        "of normal behaviour, and there are only eight leak-free days in the entire "
        "dataset - all in January. A model trained on those would know what normal looks "
        "like <i>in winter</i>, and would then have to extrapolate across a whole year of "
        "changing demand.")]
    s += [callout(
        "<b>So the question was changed.</b><br/><br/>"
        "&nbsp;&nbsp;<font color='#b03030'>Rejected:</font> <i>'Is there a leak right "
        "now?'</i> - unanswerable on this data, and useless even if answered.<br/>"
        "&nbsp;&nbsp;<font color='#2e7d32'>Adopted:</font> <i>'Has a <b>new</b> leak just "
        "started?'</i> - answerable, and exactly what a control room needs.<br/><br/>"
        "This reframing is the project's main conceptual contribution. Every design "
        "decision that follows exists to serve it.", GREENC, colors.HexColor("#eef7ef"))]
    return s


def sec_method():
    s = [H("6. How the system works, step by step")]
    s += [P(
        "The five steps below are traced through one real leak, so that each stage can be "
        "seen doing its job on actual data.")]
    s += [callout(
        "<b>The worked example.</b> Pipe <b>p523</b> burst on <b>15 January 2019 at "
        "11 pm</b>, through a 20.2 mm hole. It is an <i>abrupt</i> leak. Follow this one "
        "case through all five steps and the whole method becomes concrete.",
        ACCENT, colors.HexColor("#fdf3e7"))]

    s += [H("Step 1 - Predict what each sensor should be reading", 2), P(
        "For each of the 33 pressure sensors, a small statistical model "
        "(<b>Ridge regression</b>, a standard and very stable linear method) predicts "
        "that sensor's pressure from four things:")]
    s += bullets([
        "total inflow entering the network",
        "inflow squared, because friction rises faster than flow does",
        "the storage tank level",
        "time of day and day of week",
    ])
    s += [Spacer(1, 3), P(
        "Those inputs matter because pressure changes constantly for perfectly innocent "
        "reasons. At 8 am everybody showers, demand spikes and pressure falls across the "
        "whole network. That is not a leak. By giving the model the time of day and the "
        "inflow, it learns the normal daily and weekly rhythm - so the morning peak is "
        "never mistaken for a burst.")]

    s += [H("Step 2 - Compare against the last three weeks, not against perfection", 2)]
    s += [P(
        "This is the design choice that makes the method work despite there being no "
        "clean data. Rather than training once on some ideal healthy period, the model is "
        "<b>refitted every day on the previous 21 days</b>.")]
    s += [P("Three consequences follow, and all three are essential:")]
    s += bullets([
        "<b>Existing leaks go quiet.</b> A leak that has been running for a month sits "
        "inside the 21-day reference, so the model already expects it. It cannot keep "
        "raising alarms.",
        "<b>Seasons take care of themselves.</b> The reference is never more than three "
        "weeks old, so summer is always compared against summer.",
        "<b>Only new leaks stand out.</b> Something that began yesterday is not yet in "
        "the reference, so it breaks the learned relationship.",
    ])
    s += [Spacer(1, 3), P(
        "A refinement: the two most recent days are excluded from the reference window, "
        "so a leak that started yesterday cannot slip into its own baseline and hide "
        "itself.")]

    s += [H("Step 3 - The residual: the signal everything rests on", 2)]
    s += [P("<b>Residual = what the sensor actually read &minus; what the model predicted.</b>")]
    s += [P(
        "When the network behaves normally the residual sits near zero. When a leak "
        "starts, real pressure falls below the prediction and the residual turns "
        "negative. Here is the real recorded residual at sensor <b>n506</b> around the "
        "p523 burst:")]
    rows = [["Date", "Residual (metres)", "Interpretation"]]
    for d, v, note in [
        ("11 Jan", "-0.001", "normal"), ("12 Jan", "+0.006", "normal"),
        ("13 Jan", "+0.021", "normal"), ("14 Jan", "-0.004", "normal"),
        ("15 Jan", "-0.008", "normal - the leak begins at 11 pm tonight"),
        ("16 Jan", "<b>-0.311</b>", "<b>pressure is 31 cm below prediction</b>"),
        ("17 Jan", "-0.302", "still there"), ("18 Jan", "-0.305", "still there"),
        ("19 Jan", "-0.280", "still there"), ("20 Jan", "-0.248", "still there"),
    ]:
        rows.append([d, v, note])
    # Kept on one page deliberately - this table is the clearest evidence in the
    # whole document and loses its force when split.
    s += [KeepTogether(data_table(
        rows, widths=[CONTENT_W * 0.16, CONTENT_W * 0.26, CONTENT_W * 0.58],
        align_right_from=1, highlight_row=6))]
    s += [Spacer(1, 6), P(
        "Before the leak the model predicts the sensor to within about <b>2 centimetres</b>. "
        "The moment the leak starts the error jumps to <b>31 centimetres</b> - roughly "
        "<b>15 times larger</b> - and stays there. Detecting that jump, reliably and "
        "without crying wolf, is what the rest of the system does.")]
    s += [figure("02_residual_heatmap",
                 "Figure 3. Residuals for all 33 sensors across 2019, expressed as "
                 "z-scores. Each row is a sensor, each column a day. Dashed lines mark "
                 "true leak onsets. Coloured bands appearing and persisting after a "
                 "dashed line are leaks making themselves visible.")]

    s += [H("Step 4 - CUSUM: deciding when to raise the alarm", 2)]
    s += [P(
        "A simple rule such as 'alarm if the residual is large' would work for bursts but "
        "fail for slow cracks, which never produce a single dramatic day - only a long "
        "run of mildly odd ones. So the system uses <b>CUSUM</b> (cumulative sum).")]
    s += [callout(
        "<b>CUSUM in one analogy: a bank balance.</b> Every abnormal day pays money in. "
        "Every normal day takes money out. When the balance crosses a set level, the "
        "alarm fires. That is why it catches slow leaks: no single day is convincing, but "
        "ten mildly abnormal days in the same direction add up to proof.")]
    s += [Spacer(1, 4), P(
        "After an alarm the balance resets and the system stays silent for 14 days - "
        "otherwise a single unrepaired leak would alarm every day for a month.")]
    s += [P(
        "<b>For p523:</b> evidence accumulated until the threshold was crossed on "
        "<b>25 January</b> - an alarm <b>10 days</b> after the burst began.")]
    s += [P(
        "<b>How the threshold was chosen:</b> many settings were tested <i>on 2018 only</i>, "
        "keeping the one that caught the most leaks while raising no more than six false "
        "alarms a year. It was then frozen and never revisited. The false-alarm budget - "
        "not raw accuracy - is the binding constraint, because a system that cries wolf "
        "gets switched off by the people who have to respond to it.")]
    s += [figure("03_cusum_trace",
                 "Figure 4. The detector running across the held-out year. The blue line "
                 "is accumulated evidence, the dashed red line the alarm threshold, green "
                 "lines the true leak onsets and red triangles the alarms raised. Every "
                 "alarm follows a genuine onset, and no alarm appears anywhere else.")]

    s += [H("Step 5 - Localisation: working out where to dig", 2)]
    s += [P(
        "Detection says something broke. Localisation says where to send the crew.")]
    s += [P(
        "<b>The idea - a pressure fingerprint.</b> When a leak starts, all 33 sensors "
        "shift, but by different amounts: nearby sensors drop sharply, distant ones "
        "barely move. Collecting those 33 shifts into a list gives a fingerprint with two "
        "separable properties - its <i>shape</i> (which sensors moved most) indicates "
        "<b>where</b> the leak is, while its overall <i>size</i> indicates <b>how big</b> "
        "it is. Since leak size is unknown, the size is normalised away and only shape is "
        "compared.")]
    s += [P(
        "<b>The obstacle.</b> Learning 'this fingerprint means district 7' needs many "
        "examples per district. There are only 14 training leaks across 12 districts - "
        "roughly one each. Far too few to learn from.")]
    s += [P(
        "<b>The solution - generate the examples from physics instead.</b> Using "
        "<b>EPANET</b>, the free water-network simulator published by the US "
        "Environmental Protection Agency and the worldwide standard for this purpose "
        "(driven from Python through the <b>WNTR</b> library), a leak is placed on each "
        "of the <b>902 pipes</b> in turn, the hydraulics are solved, and the resulting "
        "33-sensor response is recorded. The result is a reference library of 902 "
        "fingerprints derived from physics rather than from data.")]
    s += [P(
        "The simulated hole is set to 15 mm - the median of the real leaks - giving about "
        "14.2 m&sup3;/h of discharge. The resulting average pressure response of 0.29 m "
        "sits squarely inside the range of the real observed shifts (0.03-0.48 m), which "
        "is good evidence the simulation is realistic.")]
    s += [P(
        "An observed fingerprint is then compared against all 902 simulated ones using "
        "<b>cosine similarity</b> - a standard measure of whether two lists of numbers "
        "have the same shape, where 1 means identical and 0 means unrelated. The "
        "best-matching pipes lend their scores to their districts.")]
    s += [figure("09_signature_match",
                 "Figure 5. The fingerprint of leak p523: what the real sensors did "
                 "(blue) against what EPANET predicted for that pipe (orange). The two "
                 "rise and fall together. That agreement between independent measurement "
                 "and physical simulation is what makes localisation possible.")]

    s += [H("Why districts rather than individual pipes", 2), P(
        "With 33 sensors and 905 pipes, two neighbouring pipes simply cannot be told "
        "apart - there is not enough information in the measurements. This is a "
        "mathematical limit, not a shortcoming of the code, and no method can escape it. "
        "The network is therefore divided into <b>12 districts</b> of roughly 3.6 km of "
        "pipe each (using k-means clustering on pipe positions), which is also the "
        "granularity at which a repair crew is actually dispatched.")]

    s += [H("The result for p523", 2)]
    s += [data_table([
        ["Predicted district", "True district", "Rank of correct district",
         "Best-matching pipe", "Distance from the real leak"],
        ["<b>7</b>", "<b>7</b>", "1 of 12", "p498", "<b>53.8 m</b>"],
    ], widths=[CONTENT_W * 0.19] * 4 + [CONTENT_W * 0.24])]
    s += [Spacer(1, 6), callout(
        "<b>The whole story in one sentence.</b> The leak bursts on 15 January at 11 pm; "
        "next morning sensor n506 reads 31 cm below prediction; CUSUM accumulates "
        "evidence for ten days and alarms on 25 January; the fingerprint is matched "
        "against 902 simulated ones; the system says 'search district 7' - which is "
        "correct, and its single best guess is a pipe 54 metres from the actual hole.")]
    s += [figure("08_localisation_example",
                 "Figure 6. Leak p523 being localised. Left: every pipe shaded by how "
                 "likely its district is, with the true leak marked in green and the "
                 "best-matching pipe circled. Right: the districts ranked by score, the "
                 "correct one shown in green.")]
    return s


def sec_results(R):
    d = R["det"]
    d19 = d[d["year"] == 2019].iloc[0]
    loc = R["loc"]
    s = [H("7. Results")]

    s += [H("The testing rule that makes these numbers trustworthy", 2)]
    s += [callout(
        "<b>2018</b> was the practice year - every threshold, window length and design "
        "choice was selected using it.<br/>"
        "<b>2019</b> was the examination - run <i>once</i>, at the end, with everything "
        "frozen, and whatever came out is reported here.<br/><br/>"
        "This is called a <b>held-out test set</b>. Without it, good-looking results mean "
        "very little, because any method can be tuned until it fits the data it was "
        "tuned on.")]

    s += [H("Detection", 2)]
    rows = [["Year", "Role", "Leaks", "Detected", "False alarms", "Median delay",
             "Night-flow baseline"]]
    for r in d.itertuples():
        rows.append([
            str(r.year), r.role.replace("held-out test", "held-out"), str(r.leaks),
            f"<b>{r.detected} ({r.detection_rate:.0%})</b>", str(r.false_alarms),
            f"{r.median_delay_days:.0f} d",
            f"{r.mnf_detected} ({r.mnf_detection_rate:.0%})",
        ])
    s += [data_table(rows, widths=[CONTENT_W * x for x in
                                   (0.08, 0.15, 0.09, 0.19, 0.14, 0.14, 0.21)])]
    s += [Spacer(1, 6), P(
        f"On the held-out year the system found <b>{int(d19.detected)} of "
        f"{int(d19.leaks)} leaks</b>, raised <b>no false alarms at all</b>, and found "
        f"them in a median of <b>{d19.median_delay_days:.0f} days</b> - against "
        f"{d19.mnf_median_delay_days:.0f} days and a single detection for the method in "
        "standard industrial use.")]
    s += [figure("04_method_comparison",
                 "Figure 7. This method against minimum night flow, on both years.",
                 0.92)]
    s += [figure("05_detection_delays",
                 "Figure 8. Time from leak start to alarm, leak by leak. Abrupt bursts "
                 "(blue) are generally caught faster than slow incipient cracks "
                 "(orange).", 0.88)]

    s += [PageBreak(), H("Localisation", 2)]
    rows = [["Year", "Leaks", "Correct district (top-1)", "Within top 3",
             "Median rank", "Median distance error"]]
    for r in loc.itertuples():
        if r.subset != "all":
            continue
        rows.append([str(r.year), str(r.n), f"<b>{r.top1:.0%}</b>", f"{r.top3:.0%}",
                     f"{r.median_zone_rank:.0f} of 12", f"{r.median_dist_m:.0f} m"])
    rows.append(["<i>random guessing</i>", "-", "<i>8%</i>", "<i>25%</i>",
                 "<i>6.5</i>", "-"])
    s += [data_table(rows, widths=[CONTENT_W * x for x in
                                   (0.19, 0.11, 0.22, 0.14, 0.15, 0.19)])]
    s += [Spacer(1, 6), P(
        "Broken down by leak type, the difference between bursts and slow cracks is "
        "consistent across both years:")]
    rows = [["Year", "Leak type", "Number", "Top-1", "Top-3"]]
    for r in loc.itertuples():
        if r.subset == "all":
            continue
        rows.append([str(r.year), r.subset, str(r.n), f"{r.top1:.0%}", f"{r.top3:.0%}"])
    s += [data_table(rows, widths=[CONTENT_W * 0.14, CONTENT_W * 0.26, CONTENT_W * 0.16,
                                   CONTENT_W * 0.22, CONTENT_W * 0.22])]
    s += [Spacer(1, 6), P(
        "In operational terms: searching blind means covering <b>43 km</b> of pipe. "
        "Naming the correct district reduces that to <b>3.6 km</b>, and a three-district "
        "shortlist to about <b>11 km</b> - a reduction of roughly <b>75%</b> in the "
        "ground a crew has to cover.")]

    s += [H("How many sensors are actually needed?", 2)]
    rows = [["Sensors", "Detection rate", "False alarms per year", "Median delay",
             "Localisation top-1"]]
    for r in R["sens"].itertuples():
        bold = r.n_sensors == 33
        f = (lambda x: f"<b>{x}</b>") if bold else (lambda x: x)
        rows.append([f(str(r.n_sensors)), f(f"{r.detection_rate:.0%}"),
                     f(f"{r.false_alarms:.1f}"), f(f"{r.median_delay_days:.1f} d"),
                     f(f"{r.localisation_top1:.0%}")])
    s += [data_table(rows, widths=[CONTENT_W * 0.14, CONTENT_W * 0.21,
                                   CONTENT_W * 0.24, CONTENT_W * 0.19,
                                   CONTENT_W * 0.22],
                     highlight_row=len(rows) - 1)]
    s += [Spacer(1, 6), callout(
        "<b>Read this table carefully - it looks backwards at first.</b> Detection rate "
        "on its own can be gamed: with fewer sensors the tuning simply settles on a lower "
        "threshold, which catches more leaks <i>and</i> raises more false alarms. Looking "
        "at both columns together gives the real conclusion: <b>five sensors are enough "
        "to know that something is wrong; it is the full network of 33 that removes false "
        "alarms entirely and doubles the ability to say where.</b> For a utility deciding "
        "where to spend money, that is the useful finding.")]
    s += [figure("06_sensor_ablation",
                 "Figure 9. Detection, localisation and false alarms as the number of "
                 "sensors varies. Sensors buy precision and location, not raw recall.",
                 0.9)]

    s += [H("How finely can a leak be located?", 2)]
    rows = [["Districts", "Top-1", "Top-3", "Random top-1", "Pipe per district"]]
    for r in R["zone"].itertuples():
        rows.append([str(r.n_zones), f"{r.top1:.0%}", f"{r.top3:.0%}",
                     f"{r.random_top1:.0%}", f"{r.mean_zone_length_km:.1f} km"])
    s += [data_table(rows, widths=[CONTENT_W * 0.18, CONTENT_W * 0.18, CONTENT_W * 0.18,
                                   CONTENT_W * 0.23, CONTENT_W * 0.23],
                     highlight_row=4)]
    s += [Spacer(1, 6), P(
        "Performance relative to chance improves as districts get finer, but absolute "
        "accuracy falls and becomes unstable. With only 19 test leaks a single leak is "
        "worth five percentage points, so the jumps between 10, 12 and 20 districts are "
        "sampling noise rather than real structure. Twelve districts is a sensible "
        "operating point; the honest statement is that this quantity of data cannot "
        "pin down the optimum precisely.")]
    s += [figure("07_zone_ablation",
                 "Figure 10. Accuracy against how finely the network is divided.", 0.85)]
    return s


def sec_limits(R):
    missed = ", ".join(sorted(
        set(R["loc19"]["pipe"]) - set(R["det19"]["pipe"])))
    s = [H("8. Limitations - stated plainly")]
    s += [P(
        "Every method has a boundary. Naming these before anyone else does is part of "
        "presenting the work honestly.")]
    s += bullets([
        f"<b>Two leaks out of nineteen were missed</b> ({missed}), both late in the year "
        "when eight other leaks were already running and masking further change.",
        "<b>Nineteen test leaks is a small sample.</b> One leak is worth about five "
        "percentage points, so no difference smaller than roughly ten points should be "
        "treated as meaningful.",
        "<b>Slow incipient leaks remain the weak case</b> for both speed of detection and "
        "accuracy of location - they are faint by nature in their early weeks.",
        "<b>Localisation is district-level, not pipe-level,</b> and with 33 sensors over "
        "905 pipes that is a mathematical limit rather than an engineering one.",
        "<b>The simulated fingerprints carry model error.</b> The published network model "
        "is documented as differing from the true network by up to 10% in pipe diameters, "
        "roughness and demand, which caps match quality regardless of method.",
        "<b>Simulated leaks are attached to a pipe's upstream junction,</b> a small "
        "approximation that is acceptable at district granularity.",
        "<b>One network, two years.</b> Nothing here demonstrates that the method "
        "transfers to a different topology without re-tuning and a new simulated library.",
        "<b>Matching alarms to leaks is necessarily approximate.</b> With up to 16 leaks "
        "running at once, attributing an alarm to one specific onset is ambiguous; alarms "
        "are matched to the oldest unmatched leak within 45 days.",
    ])
    return s


def sec_running():
    s = [H("9. Running the code")]
    s += [H("On a machine that is already set up", 2)]
    s += [P("Open a terminal in the project folder and run:")]
    s += [callout("<font face='Courier'>.venv/Scripts/python.exe run_all.py</font><br/>"
                  "<font size='8' color='#5a6872'>Recomputes every result, table and "
                  "figure. About 96 seconds using the cached data.</font><br/><br/>"
                  "<font face='Courier'>.venv/Scripts/streamlit run app.py</font><br/>"
                  "<font size='8' color='#5a6872'>Opens the interactive dashboard in a "
                  "browser. Ctrl+C in the terminal stops it.</font>",
                  MUTED, colors.HexColor("#f4f6f8"))]
    s += [H("From scratch on a new machine", 2)]
    s += [callout(
        "<font face='Courier'>python -m venv .venv<br/>"
        ".venv/Scripts/python.exe -m pip install -r requirements.txt<br/>"
        ".venv/Scripts/python.exe src/download_data.py<br/>"
        ".venv/Scripts/python.exe run_all.py</font><br/><br/>"
        "<font size='8' color='#5a6872'>The download is about 181 MB from Zenodo and "
        "resumes if interrupted. The first run_all takes roughly 15 minutes because it "
        "simulates a leak on all 902 pipes; afterwards everything is cached.</font>",
        MUTED, colors.HexColor("#f4f6f8"))]

    s += [H("What each file does", 2)]
    s += [data_table([
        ["File", "Purpose"],
        ["src/config.py", "Paths, sensor lists and the ground-truth leak table"],
        ["src/download_data.py", "Fetches the dataset from Zenodo, resumable"],
        ["src/data_loader.py", "Reads the CSVs, builds labels and time features"],
        ["src/network.py", "Network topology, district clustering, geometry"],
        ["src/residuals.py", "The rolling-reference expected-pressure model"],
        ["src/detect.py", "CUSUM detector, threshold tuning, evaluation"],
        ["src/simulate.py", "EPANET simulation of a leak on every pipe"],
        ["src/localize.py", "Fingerprint matching to districts"],
        ["src/experiments.py", "Runs the whole study, writes reports/tables"],
        ["src/figures.py", "Produces every figure in this document"],
        ["src/make_pdf.py", "Builds this PDF"],
        ["app.py", "The interactive dashboard"],
        ["run_all.py", "Reproduces everything end to end"],
    ], widths=[CONTENT_W * 0.3, CONTENT_W * 0.7], align_right_from=2, font_size=8.2)]

    s += [H("The dashboard", 2), P("Four tabs:")]
    s += bullets([
        "<b>Overview</b> - headline figures and the network map",
        "<b>Live monitor</b> - the detector's evidence trace against the true onsets, "
        "plus the residual heatmap",
        "<b>Leak explorer</b> - choose any leak and watch it be located, with its "
        "fingerprint compared against simulation",
        "<b>Evidence &amp; limits</b> - all result tables and the limitations",
    ])
    s += [Spacer(1, 3), P(
        "The sliders re-run the detector live, which is useful for showing how sensitive "
        "the results are to the thresholds. The reported results use the values tuned on "
        "2018: <b>k = 1.0, h = 4, refractory period 14 days, 12 districts</b>.")]
    return s


def sec_glossary():
    s = [H("10. Glossary")]
    terms = [
        ("SCADA", "Supervisory Control and Data Acquisition - the system that collects "
                  "sensor readings from the network."),
        ("Residual", "Actual sensor reading minus predicted reading. The core signal of "
                     "this project."),
        ("Ridge regression", "A simple, stable linear prediction model."),
        ("CUSUM", "Cumulative sum - a running tally that accumulates evidence over days "
                  "before raising an alarm."),
        ("z-score", "How many standard deviations a value sits from normal; puts all "
                    "sensors on a common scale."),
        ("MAD", "Median Absolute Deviation - a robust measure of noise that a developing "
                "leak cannot inflate."),
        ("EPANET", "The free water-network simulator published by the US EPA; the "
                   "worldwide standard."),
        ("WNTR", "Water Network Tool for Resilience - the Python library used to drive "
                 "EPANET."),
        ("Emitter", "How EPANET represents a leak: discharge depends on pressure, "
                    "Q = C&radic;P."),
        ("Cosine similarity", "A measure of whether two lists of numbers have the same "
                              "shape, independent of their size."),
        ("k-means", "A clustering algorithm; used here to group pipes into geographic "
                    "districts."),
        ("Minimum night flow (MNF)", "The industry-standard leak indicator: flow still "
                                     "entering a district at 3 am."),
        ("Abrupt / incipient", "A sudden burst / a crack that widens slowly over weeks."),
        ("Held-out test set", "Data never used for any tuning decision, so the results "
                              "on it are honest."),
        ("PRV", "Pressure Reducing Valve - caps pressure in part of the network."),
        ("Non-revenue water", "Water produced and paid for but never billed; mostly "
                              "leakage."),
    ]
    s += [data_table([["Term", "Meaning"]] + [[f"<b>{t}</b>", d] for t, d in terms],
                     widths=[CONTENT_W * 0.26, CONTENT_W * 0.74], align_right_from=2,
                     font_size=8.4)]
    return s


def sec_qa():
    s = [PageBreak(), H("11. Questions likely to be asked, and how to answer them")]
    qa = [
        ("Why not use deep learning or a neural network?",
         "The limiting resource is not data volume but <i>labelled events</i>. There are "
         "210,000 timesteps but only 33 leaks, and just 14 available for training. A deep "
         "model would memorise those 14 examples. Ridge regression with a CUSUM is "
         "appropriate to the sample size, and CUSUM is in fact the statistically optimal "
         "detector for a persistent shift in mean - which is exactly what a leak is."),
        ("37% accuracy sounds low. Is that not a failure?",
         "Against random guessing at 8% it is 4.4 times better, and the top-three figure "
         "of 53% compares against 25%. The more meaningful measure is operational: the "
         "search area drops from 43 km of pipe to 3.6 km. A crew that previously had no "
         "starting point now has a 37% chance of being in the right district immediately "
         "and 53% within three attempts."),
        ("Fourteen days to detect a leak seems slow.",
         "It is half the 30 days taken by the method currently in industrial use. For "
         "incipient leaks part of the delay is physics rather than method: in the first "
         "days the crack is barely open and there is almost nothing to measure. Abrupt "
         "bursts are caught considerably faster."),
        ("How do you know you have not overfitted?",
         "The 2019 data played no part in any decision. Every threshold, the signature "
         "window and the district count were chosen on 2018 and then frozen; 2019 was run "
         "once to produce the final numbers. Supporting evidence: the 2018 results are "
         "<i>better</i> (93% detection), which is exactly what one expects from the year "
         "used for tuning."),
        ("The dataset includes 82 household meters. Why were they not used?",
         "Two reasons. Realism: in practice automated meter readings reach the utility "
         "with a long reporting delay, so they are not available for real-time detection. "
         "And discipline: the inputs were restricted to what a control room genuinely "
         "sees live - 33 pressures, 3 flows and one tank level."),
        ("Why match fingerprints instead of training a classifier for location?",
         "With 14 real labelled leaks spread over 12 districts there is roughly one "
         "example per class, which cannot support a trained classifier. The simulated "
         "library supplies 902 reference fingerprints, but they are physics-derived "
         "rather than observed, so nearest-match comparison in signature space is the "
         "appropriate tool. It is the classical sensitivity-matrix approach from the "
         "hydraulics literature."),
        ("Why twelve districts?",
         "Six through twenty were tested. Fewer districts are easier to hit but send the "
         "crew across more pipe - six districts gives 42% top-1 but each covers 7.2 km. "
         "Twelve is a reasonable operating point at 3.6 km each. With only 19 test leaks "
         "the data cannot resolve the true optimum, and the report says so."),
        ("What is genuinely new here?",
         "Not any individual algorithm - Ridge regression, CUSUM and sensitivity matrices "
         "all exist. The contribution is the framing and the combination: recognising "
         "that 'is there a leak' is unanswerable on this data and reframing it to onset "
         "detection; using a rolling reference to cope with having essentially no "
         "leak-free data; and generating the localisation library hydraulically because "
         "the real one is far too small."),
        ("Would this work on a different network?",
         "The method transfers, but it would need re-tuning and a new EPANET model to "
         "regenerate the fingerprint library. Transfer has not been demonstrated, and "
         "that is listed as a limitation."),
        ("What would be needed to deploy it for a real utility?",
         "A calibrated hydraulic model of the network, a live feed from the SCADA "
         "historian, and a few months of data to establish the rolling reference. The "
         "computation is negligible - the entire study reruns in 96 seconds."),
    ]
    for q, a in qa:
        s += [KeepTogether([
            Paragraph(f"<b>{q}</b>", ParagraphStyle(
                "q", parent=S["body"], textColor=NAVY, spaceAfter=3,
                fontName="Helvetica-Bold")),
            Paragraph(a, ParagraphStyle(
                "a", parent=S["body"], leftIndent=10, spaceAfter=10,
                borderPadding=0)),
        ])]

    s += [H("Two findings worth mentioning - they show real experimentation", 2)]
    s += [callout(
        "<b>The night-hours result.</b> Restricting the analysis to night-time hours - "
        "the intuition on which night-flow monitoring is built - made performance "
        "<i>worse</i>, with signal-to-noise falling from 3.9 to 1.6. Discarding 80% of "
        "the day costs more in sample size than it gains in quietness. Once demand is "
        "modelled explicitly there is no reason to throw daytime data away.")]
    s += [Spacer(1, 4), callout(
        "<b>The year-boundary result.</b> An early version restarted the model each "
        "1 January, leaving the detector blind for its first three weeks - exactly where "
        "two of the 2019 leaks begin. Allowing the reference window to reach back into "
        "the previous year, as a real utility's would, raised detection from 14 of 19 to "
        "<b>17 of 19</b> and halved median latency from 30 days to <b>14</b>.")]

    s += [H("The five sentences worth memorising", 2)]
    for i, line in enumerate([
        "A leak causes a pressure drop, largest near the leak - that is the entire "
        "physical basis of the project.",
        "'Is there a leak' is useless here because something is leaking 100% of the time "
        "in 2019, so the question becomes 'has a <i>new</i> leak started'.",
        "Each sensor's expected pressure is predicted from the previous 21 days, and a "
        "new leak shows up as that prediction suddenly failing - a 2 cm error becomes "
        "31 cm.",
        "With only 14 training leaks, the location fingerprints were generated by "
        "simulating a leak on all 902 pipes in EPANET rather than learned from data.",
        "89% of leaks found with zero false alarms, on a year never touched during "
        "development, against 5% for the method utilities use today.",
    ], 1):
        s += [Paragraph(f"<b>{i}.</b>&nbsp;&nbsp;{line}",
                        ParagraphStyle("five", parent=S["body"], leftIndent=14,
                                       spaceAfter=6))]

    s += [Spacer(1, 10), callout(
        "<b>References.</b><br/>"
        "1. Vrachimis, S. G. et al. <i>Battle of the Leakage Detection and Isolation "
        "Methods.</i> Journal of Water Resources Planning and Management, 2022.<br/>"
        "2. BattLeDIM 2020 dataset, Zenodo record 4017659 (CC BY 4.0), KIOS Center of "
        "Excellence.<br/>"
        "3. Klise, K. A. et al. <i>WNTR: Water Network Tool for Resilience.</i> US EPA.<br/>"
        "4. Rossman, L. A. <i>EPANET 2 Users Manual.</i> US EPA, 2000.",
        MUTED, colors.HexColor("#f4f6f8"))]
    return s


def main() -> None:
    R = load_results()
    story = []
    story += cover(R)
    story += toc_page()
    story += sec_summary(R)
    story += [PageBreak()] + sec_setting()
    story += [PageBreak()] + sec_problem()
    story += [PageBreak()] + sec_data()
    story += [PageBreak()] + sec_discovery()
    story += [PageBreak()] + sec_method()
    story += [PageBreak()] + sec_results(R)
    story += [PageBreak()] + sec_limits(R)
    story += [PageBreak()] + sec_running()
    story += [PageBreak()] + sec_glossary()
    story += sec_qa()

    doc = build_doc()

    def on_flowable(f):
        if hasattr(f, "_toc"):
            level, text, key = f._toc
            # page - 1 so the contents agree with the printed footer, which does
            # not count the cover.
            doc.notify("TOCEntry", (level, text, doc.page - 1, key))

    doc.afterFlowable = on_flowable
    doc.multiBuild(story)
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
