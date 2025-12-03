import os
import uuid
from datetime import datetime
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Pillow is strongly recommended — this code will use it if available.
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# ------------------------------
# Helper: register unicode font
# ------------------------------
def register_unicode_font():
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont("AppFont", p))
                return "AppFont"
        except Exception:
            continue
    return None

# ------------------------------
# Markdown -> ReportLab XML (minimal)
# ------------------------------
def convert_markdown_to_xml(text: str) -> str:
    if not text:
        return ""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold pairs
    while "**" in safe:
        safe = safe.replace("**", "<b>", 1)
        if "**" in safe:
            safe = safe.replace("**", "</b>", 1)
        else:
            break
    # Italic pairs
    while "*" in safe:
        safe = safe.replace("*", "<i>", 1)
        if "*" in safe:
            safe = safe.replace("*", "</i>", 1)
        else:
            break
    safe = safe.replace("\r\n", "\n").replace("\r", "\n")
    safe = safe.replace("\n\n", "<br/><br/>")
    safe = safe.replace("\n", "<br/>")
    return safe

# ------------------------------
# Robust image scaler/loader
# ------------------------------
def create_scaled_image(path: str,
                        tmp_dir: str = "user_data/_tmp_images",
                        max_width: float = 6.8 * inch,
                        max_height: float = 8.0 * inch):
    """
    Returns a reportlab Image (Flowable) that fits within max_width x max_height
    - Uses PIL to read and optionally convert images to a temporary PNG for reliable ReportLab loading.
    - Converts pixel sizes to points using DPI if present, else assumes 96 DPI fallback.
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # If PIL available, use it to get pixel size and dpi, then save a temp PNG for RL
    if HAS_PIL:
        try:
            with PILImage.open(path) as pil:
                px_w, px_h = pil.size
                # Try to get DPI metadata; Pillow stores it as (xdpi, ydpi) in info['dpi']
                dpi_info = pil.info.get("dpi", None)
                if isinstance(dpi_info, tuple) and len(dpi_info) >= 1:
                    dpi_x = float(dpi_info[0]) if dpi_info[0] else 72.0
                    dpi_y = float(dpi_info[1]) if len(dpi_info) > 1 and dpi_info[1] else dpi_x
                elif isinstance(dpi_info, (int, float)):
                    dpi_x = dpi_y = float(dpi_info)
                else:
                    # Fallback DPI when metadata absent. 96 is common for many images; 72 is classic.
                    dpi_x = dpi_y = 96.0

                # Convert pixels -> points (1 point = 1/72 inch)
                width_pt = px_w * (72.0 / dpi_x)
                height_pt = px_h * (72.0 / dpi_y)

                # Compute scale ratio to fit box while preserving aspect
                ratio = min(max_width / width_pt, max_height / height_pt)

                # Don't blow images up too much; allow small upscale up to 1.2x
                if ratio > 1.2:
                    ratio = 1.2

                draw_w = width_pt * ratio
                draw_h = height_pt * ratio

                # Ensure positive sizes
                if draw_w <= 0 or draw_h <= 0:
                    raise ValueError("Computed non-positive image dimensions")

                # Convert to RGB & save a temporary PNG so ReportLab can read reliably on all platforms
                tmp_name = f"{uuid.uuid4().hex}.png"
                tmp_path = os.path.join(tmp_dir, tmp_name)
                # Convert palette/CMYK/etc to RGB for PDF friendliness
                rgb = pil.convert("RGBA") if pil.mode in ("RGBA", "LA") else pil.convert("RGB")
                # Save with reasonable DPI so ReportLab won't rescale oddly (store 72 DPI)
                rgb.save(tmp_path, format="PNG", dpi=(72, 72))
                # Create ReportLab Image with computed point dimensions
                rl_img = RLImage(tmp_path, width=draw_w, height=draw_h)
                return rl_img
        except Exception as e:
            # Fallback to direct RL Image with _restrictSize if PIL step fails
            try:
                rl_img = RLImage(path)
                rl_img._restrictSize(max_width, max_height)
                return rl_img
            except Exception:
                return Paragraph(f"[Image error: {e}]", getSampleStyleSheet()["BodyText"])
    else:
        # No PIL: use ReportLab's loader and restrict size
        try:
            rl_img = RLImage(path)
            rl_img._restrictSize(max_width, max_height)
            return rl_img
        except Exception as e:
            return Paragraph(f"[Image not available: {e}]", getSampleStyleSheet()["BodyText"])

# ------------------------------
# Parse doc_info.txt into summary + image/desc blocks
# ------------------------------
def parse_doc_info(path_to_doc_info: str):
    with open(path_to_doc_info, "r", encoding="utf-8") as fh:
        content = fh.read()

    # The doc_info uses blocks separated by ---; first block is overall summary
    parts = [p.strip() for p in content.split("---")]
    overall = parts[0] if parts else ""

    image_blocks = []
    for blk in parts[1:]:
        if not blk:
            continue
        lines = [l for l in blk.splitlines() if l.strip() != ""]
        if not lines:
            continue
        first = lines[0].strip()
        if first.lower().startswith("**image:") or first.lower().startswith("image:"):
            # support both bolded and plain "Image:"
            # remove leading ** and trailing **
            imgname = first.replace("**", "").split(":", 1)[1].strip()
            desc = "\n".join(lines[1:]).strip()
            image_blocks.append((imgname, desc))
        else:
            # not an image block -> skip
            continue

    return overall, image_blocks

# ------------------------------
# Main PDF generator (fixed image logic)
# ------------------------------
def generate_pdf_report(base_dir="user_data/data_meaning", output_path="user_data/final_report.pdf"):
    fontname = register_unicode_font()
    styles = getSampleStyleSheet()
    if fontname:
        styles.add(ParagraphStyle(name="TitleCenter", fontName=fontname, fontSize=26, alignment=TA_CENTER, spaceAfter=18))
        styles.add(ParagraphStyle(name="FileTitle", fontName=fontname, fontSize=18, alignment=TA_CENTER, spaceAfter=12))
        styles.add(ParagraphStyle(name="SectionHeader", fontName=fontname, fontSize=13, spaceAfter=8))
        styles.add(ParagraphStyle(name="Body", fontName=fontname, fontSize=11, leading=14))
    else:
        styles.add(ParagraphStyle(name="TitleCenter", fontSize=26, alignment=TA_CENTER, spaceAfter=18))
        styles.add(ParagraphStyle(name="FileTitle", fontSize=18, alignment=TA_CENTER, spaceAfter=12))
        styles.add(ParagraphStyle(name="SectionHeader", fontSize=13, spaceAfter=8))
        styles.add(ParagraphStyle(name="Body", fontSize=11, leading=14))

    story = []

    # COVER
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Automated Ingestion & Reporting Pipeline", styles["TitleCenter"]))
    story.append(Paragraph("Ingest • Transform • Auto-generate Executive PDFs", styles["FileTitle"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", styles["Body"]))
    story.append(PageBreak())

    if not os.path.exists(base_dir):
        print(f"[warn] base_dir {base_dir} missing")
        return

    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        doc_info_path = os.path.join(folder_path, "doc_info.txt")
        if not os.path.exists(doc_info_path):
            print(f"[warn] skipping {folder} (no doc_info.txt)")
            continue

        # File title page (folder as title)
        story.append(Paragraph(folder.replace("_", " "), styles["FileTitle"]))
        story.append(Spacer(1, 0.1 * inch))

        overall, image_blocks = parse_doc_info(doc_info_path)

        # Summary page (one page)
        story.append(Paragraph("Summary", styles["SectionHeader"]))
        story.append(Paragraph(convert_markdown_to_xml(overall), styles["Body"]))
        story.append(PageBreak())

        # One page per image + description
        for img_name, desc in image_blocks:
            image_path = os.path.join(folder_path, img_name)
            if not os.path.exists(image_path):
                # If missing, still include description page
                story.append(Paragraph("[Missing image: %s]" % img_name, styles["SectionHeader"]))
                story.append(Paragraph(convert_markdown_to_xml(desc), styles["Body"]))
                story.append(PageBreak())
                continue

            # Create scaled image flowable
            img_flow = create_scaled_image(image_path, tmp_dir=os.path.join(folder_path, "_tmp_images"))
            # Put image + space + description on one page
            story.append(KeepTogether([img_flow, Spacer(1, 0.15 * inch), Paragraph(convert_markdown_to_xml(desc), styles["Body"])]))
            story.append(PageBreak())

    # Thank you page
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("Thank you", styles["TitleCenter"]))

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(story)
    print(f"[OK] PDF generated at: {output_path}")


# Minimal main
def main():
    pass


if __name__ == "__main__":
    generate_pdf_report()
