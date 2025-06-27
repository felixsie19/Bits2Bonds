
"""
Generate a PDF report that contains a scatter-plot + trendline for
lead scores per generation.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path

def make_pdf_file():
    csv_path  = Path("../Data/lead_output_cluster.csv")
    pdf_path  = Path("../Data/report.pdf")

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["generation"], df["lead_score"],
            label="Score of lead molecule", s=20)


    z = np.polyfit(df["generation"], df["lead_score"], 1)
    p = np.poly1d(z)
    ax.plot(df["generation"], p(df["generation"]),
            "r--", linewidth=1.5, label=f"Trendline (slope={z[0]:.2f})")


    ax.set_xlabel("Generation")
    ax.set_ylabel("Performance score")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)       
    buf.seek(0)

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Lead-Molecule Performance Report", ln=True)

    pdf.set_font("Helvetica", size=10)
    pdf.ln(2)
    pdf.multi_cell(0, 5,
                "Scatter plot shows the performance score of the best molecule "
                "per generation accompanied by a least-squares trendline."
                "Top performing Molecule")

    pdf.image(buf, x=10, y=None, w=183)

    pdf.ln(95)                         


    png_path = Path("../Results/lead.png")
    pdf.image(str(png_path), x=10, y=None, w=183)  

    pdf.output(str(pdf_path))
