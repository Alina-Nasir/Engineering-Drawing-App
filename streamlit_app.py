import streamlit as st
from openai import OpenAI
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import json
import matplotlib.pyplot as plt
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
import pdfplumber
import cv2
import re
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# ---------------- PDF TEXT EXTRACTION ----------------
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---------------- GPT ANALYSIS ----------------
def build_compliance_prompt(standard_text: str, drawing_text: str) -> str:
    return f"""
You are a Senior Mechanical QA/QC Engineer working on Saudi Aramco projects.

TASK:
Check whether the DRAWING has been prepared in compliance with the STANDARD.

STANDARD DOCUMENT (Governing):
------------------------------
{standard_text}

DRAWING DOCUMENT (To be checked):
---------------------------------
{drawing_text}

REVIEW STRICTLY UNDER THE FOLLOWING SECTIONS:

1. Drawing Control & Identification
2. Applicable Codes & Standards
3. Materials of Construction
4. Dimensions, Ratings & Classes
5. Welding & NDT Requirements
6. Coating, Painting & Preservation
7. Nozzles, Manways & Orientation
8. Supports, Saddles & Structural Details
9. General Notes Compliance
10. Deviations / Non-Conformances

RULES:
- Do NOT assume compliance.
- If information is missing → mark as NON-COMPLIANT.
- Cite exact clauses or notes when possible.
- Be concise and technical.
- End with a clear verdict:
  COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT

OUTPUT FORMAT:
- Section-wise bullet points
- Final compliance verdict
"""

def analyze_changes(old_text, new_text):
    prompt = f"""
You are a Senior Mechanical / QA-QC Engineer.

Compare the OLD and NEW engineering drawings and identify changes strictly under:

1. DESIGN DATA
2. BILL OF MATERIAL (BOM)
3. GENERAL NOTES
4. WELD JOINTS
5. DRAWING TITLE & REVISION INFORMATION
6. OVERALL DRAWING REGRESSION / ENGINEERING IMPACT

Rules:
- Focus on Approval (FA) → Construction (IFC) changes
- Ignore formatting-only differences
- Use engineering terminology
- Highlight fabrication, inspection, compliance & risk impact

OLD DRAWING (FA):
-----------------
{old_text}

NEW DRAWING (IFC):
-----------------
{new_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",   # or "gpt-4.1" if available
        messages=[
            {"role": "system", "content": "You are an expert engineering document reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def check_drawing_compliance(standard_text, drawing_text):
    
    prompt = build_compliance_prompt(standard_text, drawing_text)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a strict engineering compliance reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

SECTION_LAYOUT = {
    # --- TOP LARGE DRAWING AREA ---
    "Main Elevation & Views": (0.00, 0.00, 0.765, 0.32),

    # --- RIGHT COLUMN ---
    "Design Data":           (0.74, 0.00, 1.00, 0.35),
    "Bill of Materials":     (0.74, 0.32, 1.00, 0.72),

    # --- BOTTOM LEFT ---
    "Elevator View": (0.00, 0.32, 0.56, 0.88),

    # --- BOTTOM CENTER ---
    "General Notes":         (0.54, 0.30, 0.77, 0.85),

    # --- BOTTOM RIGHT ---
    "Title Block & Revisions": (0.74, 0.81, 1.00, 1.00),

    "Weld Joints":          (0.00, 0.86, 0.27, 1.00),
}
comparison_schema = {
    "type": "object",
    "properties": {
        "comparison": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "version_A": {"type": "string"},
                    "version_B": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Same", "Changed", "Missing"]
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"]
                    }
                },
                "required": [
                    "field_name",
                    "version_A",
                    "version_B",
                    "status",
                    "confidence"
                ]
            }
        },
        "risk_summary": {"type": "string"}
    },
    "required": ["comparison", "risk_summary"]
}
bom_comparison_schema = {
    "type": "object",
    "properties": {
        "bom_comparison": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part_no": {"type": "string"},
                    "part_name": {"type": "string"},
                    "description_A": {"type": "string"},
                    "description_B": {"type": "string"},
                    "qty_A": {"type": "string"},
                    "qty_B": {"type": "string"},
                    "material_A": {"type": "string"},
                    "material_B": {"type": "string"},
                    "weight_A": {"type": "string"},
                    "weight_B": {"type": "string"},
                    "change_type": {
                        "type": "string",
                        "enum": [
                            "Unchanged",
                            "Quantity Changed",
                            "Material Changed",
                            "Weight Changed",
                            "Added",
                            "Removed"
                        ]
                    },
                    "engineering_impact": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"]
                    }
                },
                "required": [
                    "part_no",
                    "part_name",
                    "description_A",
                    "description_B",
                    "qty_A",
                    "qty_B",
                    "material_A",
                    "material_B",
                    "weight_A",
                    "weight_B",
                    "change_type",
                    "engineering_impact"
                ]
            }
        },
        "summary": {"type": "string"}
    },
    "required": ["bom_comparison", "summary"]
}

revision_comparison_schema = {
    "type": "object",
    "properties": {
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "revision": {"type": "string"},
                    "date_A": {"type": "string"},
                    "date_B": {"type": "string"},
                    "description_A": {"type": "string"},
                    "description_B": {"type": "string"},
                    "drawn_by_A": {"type": "string"},
                    "drawn_by_B": {"type": "string"},
                    "checked_by_A": {"type": "string"},
                    "checked_by_B": {"type": "string"},
                    "approved_by_A": {"type": "string"},
                    "approved_by_B": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Same", "Changed", "Added", "Removed"]
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"]
                    }
                },
                "required": [
                    "revision",
                    "status",
                    "confidence"
                ]
            }
        },
        "revision_risk_summary": {
            "type": "string"
        }
    },
    "required": ["revisions", "revision_risk_summary"]
}

general_notes_schema = {
    "type": "object",
    "properties": {
        "changed_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Exact note number as shown, e.g. 3, 3a, 3c, 14(b)"
                    },
                    "change_type": {
                        "type": "string",
                        "enum": ["Modified", "Added", "Removed"]
                    },
                    "change_summary": {
                        "type": "string",
                        "description": "Explicit description of what changed in this specific note"
                    }
                },
                "required": [
                    "note_id",
                    "change_type",
                    "change_summary"
                ]
            }
        },
        "overall_risk_comment": {
            "type": "string"
        }
    },
    "required": ["changed_notes", "overall_risk_comment"]
}

def segment_drawing(image):
    img = np.array(image)
    h, w, _ = img.shape

    segments = {}

    for section, (x1, y1, x2, y2) in SECTION_LAYOUT.items():
        x1i, y1i = int(x1 * w), int(y1 * h)
        x2i, y2i = int(x2 * w), int(y2 * h)

        segments[section] = img[y1i:y2i, x1i:x2i]

    return segments

def normalize_rotation(img):
    if img.width < img.height:
        img = img.rotate(90, expand=True)
    return img

def pdf_page_to_image(pdf_file, page_number=2, scale=2.5):
    pdf_bytes = pdf_file.getvalue()   # ✅ correct way

    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf.get_page(page_number - 1)

    bitmap = page.render(
        scale=scale,
        rotation=0
    )

    image = bitmap.to_pil()
    pdf.close()

    image = normalize_rotation(image)
    return image

def align_images_ecc(reference_img, target_img):
    ref_gray = np.array(reference_img.convert("L"))
    tgt_gray = np.array(target_img.convert("L"))

    # Resize target if needed
    if ref_gray.shape != tgt_gray.shape:
        tgt_gray = cv2.resize(tgt_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Convert to float32 (ECC requirement)
    ref_gray = ref_gray.astype(np.float32)
    tgt_gray = tgt_gray.astype(np.float32)

    # Affine transform (translation + rotation + scale)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        5000,
        1e-6
    )

    try:
        cv2.findTransformECC(
            ref_gray,
            tgt_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )
    except cv2.error:
        # Fallback: return unaligned image
        return target_img

    aligned = cv2.warpAffine(
        np.array(target_img),
        warp_matrix,
        (ref_gray.shape[1], ref_gray.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    return Image.fromarray(aligned)

def generate_clean_red_overlay(old_img, new_img):
    # 🔑 ALIGN FIRST
    new_img_aligned = align_images_ecc(old_img, new_img)

    old_gray = np.array(old_img.convert("L"))
    new_gray = np.array(new_img_aligned.convert("L"))

    diff = cv2.absdiff(old_gray, new_gray)

    _, diff_mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    overlay = np.array(new_img_aligned.convert("RGB"))
    overlay[diff_mask > 0] = [255, 0, 0]

    return overlay

def encode_image(image_input):
    """
    Accepts:
    - Streamlit UploadedFile
    - PIL Image
    - NumPy ndarray (segmented image)
    Returns:
    - base64 PNG string
    """
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)

    elif isinstance(image_input, Image.Image):
        image = image_input

    else:
        # Uploaded file or file-like object
        image = Image.open(image_input)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_data_design(img1_file,img2_file):
    img1_b64 = encode_image(img1_file)
    img2_b64 = encode_image(img2_file)

    with st.spinner("Analyzing and comparing design data"):

        response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior mechanical design reviewer and ASME Section VIII expert."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                Compare the two Design Data sheets.
                                Identify differences, reductions, or missing values.
                                Focus on safety, MAWP, pressure, temperature, materials, and compliance.
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img1_b64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img2_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "design_data_comparison",
                        "schema": comparison_schema
                    }
                }
            )


        ai_output = response.choices[0].message.content


    # st.success("Comparison Completed")
    # st.write(ai_output)
    print(ai_output)

    # ---------------------------
    # Parse AI Output
    # ---------------------------
    result = json.loads(ai_output)

    # comparison_df = pd.DataFrame(result["comparison"])

    # st.subheader("📊 Design Data Comparison")
    # st.dataframe(comparison_df, use_container_width=True)

    # st.subheader("⚠️ Engineering Risk Summary")
    # st.warning(result["risk_summary"])
    return result

def extract_data_bom(bom_img_A,bom_img_B):
    bom_img_A_b64 = encode_image(bom_img_A)
    bom_img_B_b64 = encode_image(bom_img_B)

    with st.spinner("Analyzing and comparing bill of materials..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior mechanical engineer and ASME pressure vessel BOM reviewer."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                Compare the Bill of Material (BOM) from two engineering drawings.
                                Extract BOM tables from both images.
                                Align parts using PART NO and DESCRIPTION.
                                Compare quantity, material, and weight.
                                Identify added or removed parts.
                                Assign engineering impact (Low, Medium, High).
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{bom_img_A_b64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{bom_img_B_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bom_comparison",
                        "schema": bom_comparison_schema
                    }
                }
            )




    ai_output = response.choices[0].message.content


    # st.success("Comparison Completed")
    # st.write(ai_output)
    print(ai_output)

    # ---------------------------
    # Parse AI Output
    # ---------------------------
    result = json.loads(ai_output)

    # bom_df = pd.DataFrame(result["bom_comparison"])

    # st.subheader("📦 Bill of Material Comparison")
    # st.dataframe(bom_df, use_container_width=True)

    # st.subheader("🧠 BOM Change Summary")
    # st.info(result["summary"])
    return result

def extract_data_general(gn_img_A_b64,gn_img_B_b64):
    gn_img_A_b64 = encode_image(gn_img_A)
    gn_img_B_b64 = encode_image(gn_img_B)

    with st.spinner("Analyzing and comparing revision notes..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior pressure vessel engineer and ASME Section VIII reviewer."
                },
                {
                    "role": "user",
                    "content": [
                        {
                    "type": "text",
                    "text": """
                    Compare the GENERAL NOTES sections of the two drawings.

                    CRITICAL INSTRUCTIONS:
                    - Preserve the ORIGINAL note numbering EXACTLY as written
                    - Treat sub-notes as independent notes (e.g. 3a, 3b, 3c)
                    - DO NOT renumber notes sequentially
                    - If a change occurs in a sub-note, reference the sub-note ID (e.g. Note 3c)
                    - Report ONLY notes that changed

                    For each changed note:
                    - State whether it was Added, Removed, or Modified
                    - Briefly explain what changed (1–3 sentences)

                    Do not use tables.
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{gn_img_A_b64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{gn_img_B_b64}"
                    }
                }
            ]
        }
                    ],
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "general_notes_comparison",
                            "schema": general_notes_schema
                        }
                    }
                )





    ai_output = response.choices[0].message.content


    # st.success("Comparison Completed")
    # st.write(ai_output)
    print(ai_output)

    # ---------------------------
    # Parse AI Output
    # ---------------------------
    result = json.loads(ai_output)

    # st.subheader("📝 General Notes – Exact Numbered Changes")

    # for note in result["changed_notes"]:
    #     st.markdown(
    #         f"**Note {note['note_id']} – {note['change_type']}**\n\n"
    #         f"{note['change_summary']}"
    #     )

    # st.subheader("⚠️ Overall Risk Comment")
    # st.warning(result["overall_risk_comment"])
    return result

def extract_data_revision(rev_img_A_b64, rev_img_B_b64):
    rev_img_A_b64 = encode_image(rev_img_A)
    rev_img_B_b64 = encode_image(rev_img_B)

    with st.spinner("Analyzing and comparing revision notes..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior mechanical engineer reviewing drawing revision control and contractual compliance."
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                Compare ONLY the Revision History / Revision Notes tables
                between the two drawings.

                Identify:
                - Added or removed revisions
                - Changes in revision descriptions
                - Changes in approval status (Issued for Approval vs Construction)
                - Changes in DRN / CHKD / APPD names
                - Missing or inconsistent revision sequencing

                Focus on engineering, contractual, and construction risk.
                """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{rev_img_A_b64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{rev_img_B_b64}"
                            }
                        }
                    ]
                }

                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "revision_comparison",
                        "schema": revision_comparison_schema
                    }
                }
            )




    ai_output = response.choices[0].message.content


    # st.success("Comparison Completed")
    # st.write(ai_output)
    print(ai_output)

    # ---------------------------
    # Parse AI Output
    # ---------------------------
    result = json.loads(ai_output)

    # rev_df = pd.DataFrame(result["revisions"])

    # st.subheader("📑 Revision Notes Comparison")
    # st.dataframe(rev_df, use_container_width=True)

    # st.subheader("⚠️ Revision Control Risk Summary")
    # st.warning(result["revision_risk_summary"])
    return result


def extract_plate_dims(desc):
    if not isinstance(desc, str):
        return None, None, None

    pattern = r"PLATE\s+(\d+)\s*Thk\.\s*x\s*(\d+)\s*W\s*x\s*(\d+)\s*LG"
    match = re.search(pattern, desc)

    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    return None, None, None



st.set_page_config(page_title="Design Data Comparator", layout="wide")
col1_h, col2_h, col3_h = st.columns([1, 5, 1])

with col1_h:
    st.image("sinsina_logo.jpg", width=120)

with col2_h:
    st.markdown(
        "<h2 style='text-align:center;'>Engineering Drawing Comparison</h2>"
        "<p style='text-align:center; color:gray;'>AI-Powered Design & BOM Analysis</p>",
        unsafe_allow_html=True
    )

with col3_h:
    st.image("OfficeFlow Ai-01-01.png", width=120)


col1, col2 = st.columns(2)

with col1:
    old_pdf = st.file_uploader(
        "Upload OLD Drawing",
        type="pdf"
    )

with col2:
    new_pdf = st.file_uploader(
        "Upload NEW Drawing",
        type="pdf"
    )

if old_pdf and new_pdf:

    old_img = pdf_page_to_image(old_pdf, page_number=2)
    new_img = pdf_page_to_image(new_pdf, page_number=2)

    segments_old = segment_drawing(old_img)
    segments_new = segment_drawing(new_img)

    with st.spinner("🔍 Analyzing drawing revisions..."):
        old_text = extract_pdf_text(old_pdf)
        new_text = extract_pdf_text(new_pdf)
        standard_text = extract_pdf_text('SD-8100-13513-0001_0F2_001_LPPT.pdf')
        report = check_drawing_compliance(standard_text, new_text)

    st.success("✅ Analysis completed")

    st.subheader("📊 Engineering Compliance Report")
    st.markdown(report)

    st.download_button(
        "📥 Download Compliance Report",
        data=report,
        file_name="drawing_revision_comparison.txt",
        mime="text/plain"
    )

    overlay_img = generate_clean_red_overlay(old_img, new_img)
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.image(old_img, caption="Original Drawing – Page 2", use_container_width=True)

    # with col2:
    #     st.image(new_img, caption="Modified Version – Page 2", use_container_width=True)

    # with col3:
    st.image(overlay_img,caption="🔴 Changed Elements Highlighted (Modified Vs Original)",use_container_width=True)

    # cols = st.columns(3)
    # i = 0

    # for section, img in segments_old.items():
    #     with cols[i % 3]:
    #         st.image(img, caption=section, use_container_width=True)
    #     i += 1

    # new_cols = st.columns(3)
    # i = 0
    # st.write('New version Segments')
    # for section, img in segments_new.items():
    #     with new_cols[i % 3]:
    #         st.image(img, caption=section, use_container_width=True)
    #     i += 1
    img_1_file = segments_old["Design Data"]
    img_2_file = segments_new["Design Data"]
    bom_img_A = segments_old["Bill of Materials"]
    bom_img_B = segments_new["Bill of Materials"]
    rev_img_A = segments_old["Title Block & Revisions"]
    rev_img_B = segments_new["Title Block & Revisions"]
    gn_img_A = segments_old["General Notes"]
    gn_img_B = segments_old["General Notes"]
    result1 = extract_data_design(img_1_file,img_2_file)
    result2 = extract_data_bom(bom_img_A,bom_img_B)
    result3 = extract_data_general(gn_img_A,gn_img_B)
    result4 = extract_data_revision(rev_img_A,rev_img_B)
    tab_design, tab_bom, tab_general, tab_revision = st.tabs(
    ["📐 Design Data", "📦 BOM", "📝 General Notes", "🕘 Revisions"]
    )
    with tab_design:
        st.subheader("Design Data Comparison")
        comparison_df = pd.DataFrame(result1["comparison"])

        st.subheader("📊 Design Data Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("⚠️ Engineering Risk Summary")
        st.warning(result1["risk_summary"])

    with tab_bom:
        st.subheader("Bill of Materials Comparison")
        bom_df = pd.DataFrame(result2["bom_comparison"])

        st.subheader("📦 Bill of Material Comparison")
        st.dataframe(bom_df, use_container_width=True)

        st.subheader("🧠 BOM Change Summary")
        st.info(result2["summary"])
        plate_rows = []

        bom_df[["Thickness (mm)", "Width (mm)", "Length (mm)"]] = (
            bom_df["description_A"]
            .apply(lambda x: pd.Series(extract_plate_dims(x)))
        )


        bom_df["qty_A"] = pd.to_numeric(bom_df["qty_A"], errors="coerce").fillna(1)
        bom_df["qty_B"] = pd.to_numeric(bom_df["qty_B"], errors="coerce").fillna(1)
        bom_df["Total Length A (mm)"] = bom_df["Length (mm)"] * bom_df["qty_A"]
        bom_df["Total Length B (mm)"] = bom_df["Length (mm)"] * bom_df["qty_B"]
        plates_df = bom_df.dropna(subset=["Thickness (mm)", "Length (mm)"])

        summary_A = (
            plates_df
            .groupby(
                ["Thickness (mm)", "Width (mm)", "material_A"],
                as_index=False
            )
            .agg({"Total Length A (mm)": "sum"})
        )

        summary_A["Total Length A (m)"] = summary_A["Total Length A (mm)"] / 1000
        summary_B = (
            plates_df
                    .groupby(
                        ["Thickness (mm)", "Width (mm)", "material_B"],
                        as_index=False
                    )
                    .agg({"Total Length B (mm)": "sum"})
                )

        summary_B["Total Length B (m)"] = summary_B["Total Length B (mm)"] / 1000
        
        st.subheader("📐 Plate Material Summary — Drawing A")
        st.dataframe(summary_A, use_container_width=True)

        st.subheader("📐 Plate Material Summary — Drawing B")
        st.dataframe(summary_B, use_container_width=True)




    with tab_general:
        st.subheader("General Notes Comparison")
        for note in result3["changed_notes"]:
            st.markdown(
                f"**Note {note['note_id']} – {note['change_type']}**\n\n"
                f"{note['change_summary']}"
            )
        st.subheader("⚠️ Overall Risk Comment")
        st.warning(result3["overall_risk_comment"])

    with tab_revision:
        st.subheader("Revision History Comparison")
        rev_df = pd.DataFrame(result4["revisions"])

        st.subheader("📑 Revision Notes Comparison")
        st.dataframe(rev_df, use_container_width=True)

        st.subheader("⚠️ Revision Control Risk Summary")
        st.warning(result4["revision_risk_summary"])


    

