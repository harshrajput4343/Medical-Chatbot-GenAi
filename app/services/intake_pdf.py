from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime


def generate_intake_pdf(intake) -> BytesIO:
    """Generate a clean, printable A4 clinical PDF from a PatientIntake record."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()

    # --- Custom Styles (clinical black-on-white, no color backgrounds) ---
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=4,
        textColor=colors.HexColor("#111827"),
        fontName="Helvetica-Bold",
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=20,
    )

    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#111827"),
        spaceBefore=18,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=6,
    )

    label_style = ParagraphStyle(
        "LabelStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#374151"),
        fontName="Helvetica-Bold",
    )

    value_style = ParagraphStyle(
        "ValueStyle",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#111827"),
    )

    footer_style = ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#9ca3af"),
        spaceBefore=30,
    )

    elements = []

    # --- Header ---
    elements.append(Paragraph("MedBot AI", title_style))
    elements.append(Paragraph("Patient Pre-Consultation Report", subtitle_style))
    elements.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.HexColor("#d1d5db"),
        spaceAfter=18, spaceBefore=4,
    ))

    # --- Patient Info Grid ---
    created_date = intake.created_at.strftime("%B %d, %Y — %I:%M %p") if intake.created_at else datetime.utcnow().strftime("%B %d, %Y — %I:%M %p")

    info_data = [
        [
            Paragraph("<b>Patient Name:</b>", label_style),
            Paragraph(str(intake.name), value_style),
            Paragraph("<b>Date:</b>", label_style),
            Paragraph(created_date, value_style),
        ],
        [
            Paragraph("<b>Age:</b>", label_style),
            Paragraph(str(intake.age), value_style),
            Paragraph("<b>Sex:</b>", label_style),
            Paragraph(str(intake.sex), value_style),
        ],
    ]

    info_table = Table(info_data, colWidths=[3.5 * cm, 5.2 * cm, 2.5 * cm, 5.5 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d1d5db")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(info_table)

    # --- Clinical Sections ---
    sections = [
        ("Chief Complaint", intake.chief_complaint),
        ("Duration of Issue", intake.duration),
        ("Past Symptoms", intake.past_symptoms or "None reported"),
        ("Previous Medications", intake.previous_medications or "None reported"),
        ("Previous Tests", intake.previous_tests or "None reported"),
        ("Previous Hospital Visit", "Yes" if intake.previous_visit else "No"),
    ]

    for heading, content in sections:
        elements.append(Paragraph(heading, section_header_style))
        elements.append(HRFlowable(
            width="100%", thickness=0.75,
            color=colors.HexColor("#d1d5db"),
            spaceAfter=8,
        ))
        elements.append(Paragraph(str(content), body_style))

    # --- Footer ---
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#d1d5db"),
        spaceAfter=10,
    ))
    elements.append(Paragraph(
        f"Generated on {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')} — For Doctor Use Only",
        footer_style,
    ))
    elements.append(Paragraph(
        "This is a system-generated pre-consultation report. It does not constitute medical advice.",
        footer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
