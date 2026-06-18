"""Build the PDF report for the Demand Forecasting project.
Run after 01_analyse.py.  Output: report/Demand_Forecasting_Report.pdf
"""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

OUT = Path("outputs"); REP = Path("report"); REP.mkdir(exist_ok=True)
m = json.load(open(OUT / "metrics.json"))
INK = colors.HexColor("#1d3a34"); TEAL = colors.HexColor("#2a9d8f")
GOLD = colors.HexColor("#e9b949"); CORAL = colors.HexColor("#e76f51")
LINE = colors.HexColor("#e3ece9")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=INK, fontSize=22, spaceAfter=6)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=11, textColor=colors.HexColor("#5d736e"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=INK, fontSize=15, spaceBefore=14, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10.3, leading=15, spaceAfter=6)
NOTE = ParagraphStyle("NOTE", parent=BODY, leftIndent=8, textColor=colors.HexColor("#243b37"),
                      backColor=colors.HexColor("#f1f7f5"), borderPadding=6, spaceAfter=8)
MEAN = ParagraphStyle("MEAN", parent=NOTE, backColor=colors.HexColor("#fdf6e9"))
REC = ParagraphStyle("REC", parent=NOTE, backColor=colors.HexColor("#fbeee9"))
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=9, textColor=colors.HexColor("#6a807b"))

fc = m["forecast"]["models"]; iv = m["inventory"]; ed = m["eda"]
story = []


def fig(name, w=16):
    img = Image(str(OUT / name)); ratio = img.imageHeight / img.imageWidth
    img.drawWidth = w * cm; img.drawHeight = w * ratio * cm
    return img


story += [Paragraph("Demand Forecasting and Inventory Optimisation", H1),
          Paragraph("Forecasting daily sales for 500 store-item combinations, then "
                    "turning the forecast into smarter stocking decisions. "
                    "Kingsley Amegah, Data Scientist.", SUB),
          Spacer(1, 10)]

kpis = [["913,000", "11.4%", "about 50%", "13%"],
        ["daily records (2013-2017)", "forecast error (WMAPE)",
         "less safety stock", "less inventory, same service"]]
t = Table(kpis, colWidths=[4 * cm] * 4)
t.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, 0), 15), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("TEXTCOLOR", (0, 0), (-1, 0), TEAL), ("FONTSIZE", (0, 1), (-1, 1), 8),
    ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#5d736e")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE), ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
story += [t, Spacer(1, 12)]

story += [Paragraph("1. The business problem", H2),
          Paragraph("Every retailer faces the same tension. Hold too little stock and "
                    "you get stockouts, meaning lost sales and unhappy customers. Hold "
                    "too much and you tie up cash, fill the warehouse and risk markdowns "
                    "and waste. The lever that controls both is how accurately demand can "
                    "be predicted, and how well that prediction is turned into reorder "
                    "decisions. Using five years of real daily sales for 10 stores and 50 "
                    "items, this project learns the demand patterns, builds and tests a "
                    "forecasting model, and converts the forecast into an inventory policy "
                    "whose value is measured in a simulation.", BODY)]

story += [Paragraph("2. How demand behaves", H2), fig("fig1_trend.png"),
          Paragraph("<b>What this shows.</b> Total units sold each day, with a 30-day "
                    f"average. Demand grows about {ed['avg_annual_growth_pct']}% per year "
                    "with a strong yearly wave.", NOTE),
          Paragraph("<b>What it means.</b> A forecast must capture the trend and the "
                    "seasonal shape, not just repeat last week, or it will run low in the "
                    "busy season and overstock in the quiet one.", MEAN),
          fig("fig2_seasonality.png"),
          Paragraph("<b>What this shows.</b> The weekly and monthly patterns. Weekends "
                    f"sell about {ed['weekend_uplift_pct']}% more than weekdays and demand "
                    "peaks in summer.", NOTE),
          Paragraph("<b>What it means.</b> These two rhythms are the backbone of the "
                    "forecast and should guide staffing and deliveries.", MEAN)]

story += [PageBreak(), Paragraph("3. Forecasting: testing four models honestly", H2),
          Paragraph("To trust a forecast it must be tested on unseen data. I held out the "
                    "last 90 days and compared four models. Every feature the "
                    "machine-learning model uses is at least 90 days old, so it never sees "
                    "into the period it predicts.", BODY),
          fig("fig4_model_comparison.png", 13)]
tbl = [["Model", "WMAPE", "MAE"]]
for n in ["Naive", "Seasonal naive", "ETS", "Global ML"]:
    tbl.append([n, f"{fc[n]['WMAPE_pct']}%", f"{fc[n]['MAE']}"])
mt = Table(tbl, colWidths=[6 * cm, 4 * cm, 4 * cm])
mt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#eaf5f2")),
    ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
story += [mt, Spacer(1, 6),
          Paragraph("<b>What it means.</b> The machine-learning model wins clearly at "
                    f"{fc['Global ML']['WMAPE_pct']}% error, about "
                    f"{m['forecast']['ml_vs_snaive_error_reduction_pct']:.0f}% more "
                    "accurate than the strong seasonal-naive benchmark. Learning from all "
                    "500 series at once lets it borrow patterns a single-series model "
                    "misses.", MEAN),
          fig("fig5_forecast_example.png"),
          Paragraph("<b>What this shows.</b> One store-item; the shaded area is the 90-day "
                    "test window the model never saw. The forecast tracks both the weekly "
                    "swing and the December slowdown.", NOTE)]

story += [PageBreak(), Paragraph("4. From forecast to inventory decisions", H2),
          Paragraph("A forecast creates value only when it changes a decision. Each "
                    "forecast becomes a weekly order-up-to policy: every week, top each "
                    "store-item up to a level covering expected demand over the reorder "
                    "cycle plus a safety-stock buffer sized for 95% service. The buffer "
                    "depends directly on how uncertain the forecast is.", BODY),
          fig("fig6_safety_stock.png", 11),
          Paragraph("<b>What it means.</b> Because the model already explains the weekly "
                    "and seasonal swings, the leftover uncertainty is far smaller, so the "
                    f"buffer can be about {iv['safety_stock_reduction_pct']:.0f}% smaller "
                    "for the same service promise.", MEAN),
          fig("fig7_simulation.png"),
          Paragraph("<b>What this shows.</b> A day-by-day simulation of the weekly reorder "
                    "policy across all 500 store-items, run from historical variability "
                    "(grey) versus from the forecast (teal).", NOTE),
          Paragraph("<b>What it means.</b> Both clear the 95% target, but the "
                    f"forecast-driven policy does it with {iv['holding_savings_pct']:.0f}% "
                    "less inventory on hand. The historical policy reaches "
                    f"{iv['sim_fill_rate_historical']:.1f}% only by overstocking, capital "
                    "that earns nothing.", MEAN),
          fig("fig8_pareto.png", 11),
          Paragraph("<b>What it means.</b> The top 20% of store-items drive about "
                    f"{iv['top20pct_revenue_share']:.0f}% of revenue, so forecasting and "
                    "buffering effort should be tightest there.", MEAN)]

story += [PageBreak(), Paragraph("5. Recommendations", H2),
          Paragraph("<b>1. Adopt the machine-learning forecast for replenishment</b>, about "
                    "28% more accurate than current practice and scalable to all series "
                    "from one model.", REC),
          Paragraph("<b>2. Re-size safety stock from forecast error, not raw demand "
                    "swings</b>, roughly halving the buffer and cutting inventory about "
                    "13% while still meeting the 95% service target.", REC),
          Paragraph("<b>3. Differentiate by importance</b>, holding the tightest service "
                    "on the top 20% of store-items that drive a third of revenue.", REC),
          Paragraph("<b>4. Validate live before full rollout</b>, piloting on a sample of "
                    "stores, tracking realised service and inventory for a quarter, then "
                    "scaling and re-training as demand shifts.", REC),
          Paragraph("6. Method and limitations", H2),
          Paragraph("Data: Store Item Demand Forecasting Challenge (Kaggle), 10 stores, 50 "
                    "items, daily sales 2013-2017, no missing values. Forecasting: a global "
                    "HistGradientBoostingRegressor with calendar features and lags of 90+ "
                    "days, benchmarked against naive, seasonal-naive and ETS on a 90-day "
                    "holdout. Inventory: a periodic-review order-up-to policy, 7-day lead "
                    "time, 95% target, evaluated in a day-by-day simulation. The cost "
                    "figures use assumed unit, holding and order costs; the percentage "
                    "improvements do not depend on those assumptions. Lead time is treated "
                    "as fixed; real lead-time variability would call for a slightly larger "
                    "buffer.", BODY),
          Spacer(1, 8),
          Paragraph("Built in Python (pandas, scikit-learn, statsmodels) from open data. "
                    "Code: github.com/Kingsley-amg/demand-forecasting", CAP)]

doc = SimpleDocTemplate(str(REP / "Demand_Forecasting_Report.pdf"), pagesize=A4,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm)
doc.build(story)
print("Wrote report/Demand_Forecasting_Report.pdf")
