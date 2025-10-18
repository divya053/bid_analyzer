# # Streamlit RFP Analyzer (docTR OCR + GPT-4 + .env + Token-safe Top 10 Chunks)
# # ----------------------------------------------------------------------------
# # Upload RFP + optional company PDFs (IKIO, METCO, Sunsprint)
# # OCR via docTR, GPT-4 analysis (limited to 10 most relevant text chunks)
# # Shows only applicable companies per mapping (scores & Go/No-Go)
# # ----------------------------------------------------------------------------

# import os
# import json
# import tempfile
# import numpy as np
# import streamlit as st
# from PIL import Image
# from dotenv import load_dotenv
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from openai import OpenAI
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from io import BytesIO
# try:
#     from docx import Document as DocxDocument
# except Exception:
#     DocxDocument = None

# # --------------- Setup ---------------
# st.set_page_config(page_title="RFP Analyzer (GPT-4 + docTR)", layout="wide")
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("❌ OPENAI_API_KEY not found in .env file.")
#     st.stop()

# client = OpenAI(api_key=api_key)

# # --------------- Constants ---------------
# MAPPING_TEXT = """
# - Lighting (Supply +Subsitution Allowed) → IKIO
# - Lighting (Supply+Installation+Subsitution Not allowed) → Sunsprint Engineering or METCO Engineering
# - Lighting (Supply + Installation +Subsitution Allowed) → IKIO or METCO Engineering or Sunsprint Engineering
# - HVAC → Sunsprint Engineering or METCO Engineering
# - Solar PV → Sunsprint Engineering or METCO Engineering
# - Lighting (Installation) → Sunsprint Engineering or METCO Engineering
# - Water Management → Sunsprint Engineering or METCO Engineering
# - Building Envelope → Sunsprint Engineering or METCO Engineering
# - Construction → Sunsprint Engineering or METCO Engineering
# - ESCO → Sunsprint Engineering or METCO Engineering
# - Emergency Generator → Sunsprint Engineering or METCO Engineering
# """.strip()

# # Paste the FULL content of BID Evaluation Steps.docx below.
# # This string will be used as the STRICT system prompt by default (no upload needed).
# EVAL_STEPS_TEXT = """
# <<PASTE THE FULL CONTENT OF BID Evaluation Steps.docx HERE>>
# """.strip()

# # Primary evaluation questions used for scoring (Yes/Partial/No)
# PRIMARY_QUESTIONS = [
#     "What is the project state?",
#     "Is company/contractor/subcontractor registered or licensed to work in the project state?",
#     "What are the minimum qualifications?",
#     "Is any SBE, WBE, MBE, HUB requirement in project, and can company meet this?",
#     "Is Project scope clear and achievable with our company's capabilities?",
#     "Is any Labor relations/Union requirements? If so, are these requirements understood and manageable?",
#     "Is Site investigation conducted or required?",
#     "Is any adherence to 'BABA/BAA' or other domestic content rules is required in the BID ?",
#     "Is it possible for the company to fulfil the BABA/BAA complaince ?",
#     "Is any Contractual Terms & Conditions (indemnity, liability, payment) are required in the BID?",
#     "Is any Required bonds (bid, performance, payment) and insurances in the BID can be secured by the company?",
# ]

# # --------------- OCR Extraction ---------------
# @st.cache_data(show_spinner=False)
# def extract_text(file_bytes, filename):
#     predictor = ocr_predictor(pretrained=True)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
#         tmp.write(file_bytes)
#         tmp_path = tmp.name
#     try:
#         if filename.lower().endswith(".pdf"):
#             doc = DocumentFile.from_pdf(tmp_path)
#         else:
#             img = Image.open(tmp_path).convert("RGB")
#             doc = DocumentFile.from_images([img])
#         result = predictor(doc)
#         export = result.export()
#         text = "\n".join(
#             [
#                 " ".join(w.get("value", "") for w in line.get("words", []))
#                 for page in export.get("pages", [])
#                 for block in page.get("blocks", [])
#                 for line in block.get("lines", [])
#             ]
#         )
#         return text
#     finally:
#         os.remove(tmp_path)

# # --------------- GPT-4 Analysis (Token-Safe) ---------------
# def analyze_with_gpt(rfp_text, company_docs, eval_steps: list[str] | None = None):
#     # Chunk RFP text
#     chunks = [rfp_text[i:i+1500] for i in range(0, len(rfp_text), 1500)]
#     if len(chunks) > 10:
#         # TF-IDF similarity scoring for top chunks
#         keywords = ["scope", "work", "bid", "lighting", "hvac", "installation",
#                     "requirements", "solar", "construction", "license",
#                     "proposal", "project", "compliance", "timeline", "schedule"]
#         vectorizer = TfidfVectorizer(stop_words="english")
#         tfidf = vectorizer.fit_transform(chunks + [" ".join(keywords)])
#         sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
#         top_idx = np.argsort(sims)[::-1][:10]
#         chunks = [chunks[i] for i in top_idx]
#     selected_text = "\n\n".join(chunks)

#     trimmed_docs = {k: v[:60000] for k, v in company_docs.items() if v and v.strip()}

#     # Use uploaded steps if provided, otherwise the embedded EVAL_STEPS_TEXT
#     _steps_source = eval_steps if (eval_steps and any(str(s).strip() for s in eval_steps)) else (EVAL_STEPS_TEXT.splitlines() if EVAL_STEPS_TEXT else [])
#     steps_text = "\n   - ".join([s for s in _steps_source if str(s).strip()])
#     system_prompt = f"""
# You are an expert in analyzing BID/RFP/RFQ/RFI documents.

# You are given:
# 1) An RFP/BID document (OCR'ed text).
# 2) Optional capability documents for IKIO, METCO Engineering, and Sunsprint Engineering.

# Follow these rules and return **valid JSON ONLY** with the schema below.

# GOALS:
# A) Identify work profile(s) in the RFP and recommend company/companies per the mapping (best 2 if applicable).
# B) Produce a concise summary including:
#    - Scope of Work with key project info (state, ALL dates, location, etc.)
#    - Key requirements (codes, compliances, licenses, bonds, insurance, etc.)
#    - Specific recommendations to fulfill BEFORE bidding.
# C) For each **relevant company only** (per mapping), provide:
#    - relevance_score (0–100)
#    - strengths
#    - weaknesses
#    - why_recommended
#    - why_not_others
#    - fit_summary
# D) Evaluate the PRIMARY QUESTIONS below for each relevant company and return JSON objects per question with:
#    - question, answer (Yes|Partial|No|Unknown), justification (<=25 words, cite RFP or Company evidence), score (Yes=1.0, Partial=0.5, No/Unknown=0.0).
#    - Add items with answer Yes to strengths; items with No/Unknown to weaknesses (short phrasing).
#    PRIMARY QUESTIONS:
#    - {"\n   - ".join(PRIMARY_QUESTIONS)}
# E) Provide Go / No-Go (2–3 line rationale) **only for the top recommended company**.
# F) STRICTLY evaluate every BID Evaluation Step provided (from the uploaded DOCX). For each step, return:
#    - step (verbatim), status (Yes|Partial|No|Unknown), justification (<=25 words), action_required (<=20 words if not Yes).

# COMPANY MAPPING (HARD CONSTRAINT):
# {MAPPING_TEXT}

# ⚠️ IMPORTANT RULE:
# Only score companies relevant to the detected work profile(s).
# Example: if Lighting (Supply +Substitution Allowed) → IKIO, only score IKIO, skip METCO & Sunsprint.

# # Answers of these questions to reflect in strengths/weaknesses with scoring to each questions
# # If Yes/clear/achievable → add to strengths; if No/Unknown/not achievable → add to weaknesses.
# #  What is the project state?
# #  Is company/contractor/subcontractor registered or licensed to work in the project state?
# # What are the minimum qualifications? 
# # Is any SBE, WBE, MBE, HUB requirement in project, and can company meet this?
# # Is Project scope clear and achievable with our company's capabilities?
# # Is any Labor relations/Union requirements? If so, are these requirements understood and manageable?
# # Is Site investigation conducted or required? 
# # Is any adherence to 'BABA/BAA' or other domestic content rules is required in the BID ? 
# # Is it possible for the company to fulfil the BABA/BAA complaince ?
# # Is any Contractual Terms & Conditions (indemnity, liability, payment) are required in the BID? 
# # Is any Required bonds (bid, performance, payment) and insurances in the BID can be secured by the company?


# SCHEMA (JSON ONLY):
# {{
#   "work_profile_recommendations": [
#     {{
#       "work_profile": str,
#       "recommended_companies": ["IKIO"|"METCO Engineering"|"Sunsprint Engineering", ...],
#       "evidence": [str]
#     }}
#   ],
#   "summary": {{
#     "scope_of_work": str,
#     "project_info_points": [str],
#     "key_requirements": [str],
#     "specific_recommendations_before_bidding": [str]
#   }},
#   "company_evaluation": [
#     {{
#       "company": str,
#       "question_assessments": [
#         {{
#           "question": str,
#           "answer": "Yes"|"Partial"|"No"|"Unknown",
#           "justification": str,
#           "score": float
#         }}
#       ],
#       "strengths": [str],
#       "weaknesses": [str],
#       "relevance_score": int,
#       "why_recommended": str,
#       "why_not_others": str,
#       "fit_summary": str
#     }}
#   ],
#   "best_recommendation": {{
#     "company": str,
#     "go_no_go": "Go"|"Go (conditional)"|"No-Go",
#     "go_no_go_rationale_2_3_lines": str,
#     "justification": str
#   }},
#   "bid_evaluation_steps": [
#     {{
#       "step": str,
#       "status": "Yes"|"Partial"|"No"|"Unknown",
#       "justification": str,
#       "action_required": str
#     }}
#   ]
# }}

# NOTES:
# - Keep output ≤ 1000 words.
# - Base evaluations on mapping + company docs.
# - Output JSON only.
#  - Use the following PRIMARY QUESTIONS for scoring (allocate points equally across all questions):
#    - {" | ".join(PRIMARY_QUESTIONS)}
#  - Scoring rule per question: Yes=1.0, Partial=0.5, No=0.0, Unknown=0.0.
#  - relevance_score = round(100 * (sum(question scores) / number_of_questions)). Provide concise justifications tied to RFP/company evidence.
#  - If provided, also follow these BID Evaluation Steps and reflect them in justifications and recommendations:
#    - {steps_text}
#  - You MUST include every provided BID Evaluation Step in the output array 'bid_evaluation_steps'. Do not omit any step.
# """.strip()
#     # If BID Evaluation Steps were provided, override prompt to use ONLY that content strictly
#     # Keep the full schema/instructions; steps are already embedded above

#     # Build user prompt
#     user_prompt = "RFP SELECTED CHUNKS (Top 10):\n" + selected_text + "\n\n"
#     if trimmed_docs:
#         for name, doc in trimmed_docs.items():
#             user_prompt += f"{name} CAPABILITY DOCUMENT START\n{doc}\n{name} CAPABILITY DOCUMENT END\n\n"
#     # Ensure assistant sees 'json' keyword to satisfy response_format requirement
#     user_prompt += "\nReturn JSON only."

#     # GPT-4 call with strict JSON response
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         temperature=0.2,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#     )

#     raw = response.choices[0].message.content or ""
#     try:
#         return json.loads(raw)
#     except Exception:
#         # Fallback: try to extract JSON substring
#         try:
#             start, end = raw.find("{"), raw.rfind("}") + 1
#             return json.loads(raw[start:end])
#         except Exception:
#             # Graceful minimal structure to avoid crashing UI
#             return {
#                 "summary": {
#                     "scope_of_work": raw[:1000]
#                 },
#                 "company_evaluation": [],
#                 "best_recommendation": {
#                     "company": "N/A",
#                     "go_no_go": "N/A",
#                     "go_no_go_rationale_2_3_lines": "",
#                     "justification": ""
#                 },
#                 "bid_evaluation_steps": []
#             }

# # --------------- Streamlit UI ---------------
# st.title("📄 RFP Analyzer — Token-Safe Version")

# rfp_file = st.file_uploader("📤 Upload RFP/BID/RFQ/RFI Document", type=["pdf", "png", "jpg", "jpeg", "tiff"])

# st.subheader("🏢 Upload Company Capability Documents (optional)")
# c1, c2, c3 = st.columns(3)
# with c1:
#     ikio_file = st.file_uploader("IKIO Profile", type=["pdf", "png", "jpg", "jpeg"], key="ikio")
# with c2:
#     metco_file = st.file_uploader("METCO Engineering Profile", type=["pdf", "png", "jpg", "jpeg"], key="metco")
# with c3:
#     suns_file = st.file_uploader("Sunsprint Engineering Profile", type=["pdf", "png", "jpg", "jpeg"], key="suns")

# if rfp_file:
#     st.info("🔍 Extracting text from RFP...")
#     rfp_text = extract_text(rfp_file.read(), rfp_file.name)
#     if not rfp_text.strip():
#         st.error("No text extracted.")
#         st.stop()
#     st.success("✅ RFP OCR complete.")
#     with st.expander("Preview Extracted RFP Text", expanded=False):
#         st.text_area("RFP Preview", rfp_text[:5000], height=200)

#     company_docs = {}
#     if ikio_file:
#         st.info("Reading IKIO profile…")
#         company_docs["IKIO"] = extract_text(ikio_file.read(), ikio_file.name)
#     if metco_file:
#         st.info("Reading METCO profile…")
#         company_docs["METCO Engineering"] = extract_text(metco_file.read(), metco_file.name)
#     if suns_file:
#         st.info("Reading Sunsprint profile…")
#         company_docs["Sunsprint Engineering"] = extract_text(suns_file.read(), suns_file.name)

#     # Optional: Upload BID Evaluation Steps (DOCX) to guide analysis
#     st.subheader("📝 BID Evaluation Steps (embedded by default; optional DOCX override)")
#     steps_file = st.file_uploader("BID Evaluation Steps.docx", type=["docx"], key="bid_steps")
#     eval_steps = []
#     if steps_file and DocxDocument is not None:
#         try:
#             docx = DocxDocument(BytesIO(steps_file.read()))
#             # Extract non-empty lines as steps
#             for p in docx.paragraphs:
#                 line = (p.text or "").strip()
#                 if line:
#                     eval_steps.append(line)
#         except Exception:
#             eval_steps = []

#     if st.button("🚀 Analyze with GPT-4"):
#         with st.spinner("Analyzing top 10 RFP sections with GPT-4…"):
#             result = analyze_with_gpt(rfp_text, company_docs, eval_steps=eval_steps)

#         if result:
#             # Work profile recommendations
#             if result.get("work_profile_recommendations"):
#                 st.header("🧭 Work Profile → Recommended Companies")
#                 for rec in result["work_profile_recommendations"]:
#                     st.markdown(f"**{rec.get('work_profile','')}** → {', '.join(rec.get('recommended_companies', []))}")
#                     if rec.get("evidence"):
#                         with st.expander("Evidence from RFP", expanded=False):
#                             for e in rec["evidence"]:
#                                 st.markdown(f"- {e}")

#             # Summary
#             st.header("📘 Summary of RFP")
#             summary = result.get("summary", {}) or {}
#             if summary.get("scope_of_work"):
#                 st.markdown("### Scope of Work")
#                 st.write(summary["scope_of_work"])
#             if summary.get("project_info_points"):
#                 st.markdown("### Project Information (Key Points)")
#                 for p in summary["project_info_points"]:
#                     st.markdown(f"- {p}")
#             if summary.get("key_requirements"):
#                 st.markdown("### Key Requirements")
#                 for r in summary["key_requirements"]:
#                     st.markdown(f"- {r}")
#             if summary.get("specific_recommendations_before_bidding"):
#                 st.markdown("### Recommendations Before Bidding")
#                 for r in summary["specific_recommendations_before_bidding"]:
#                     st.markdown(f"- {r}")

#             # Company Evaluation
#             st.header("🏢 Company Evaluation (Applicable Only)")
#             for comp in result.get("company_evaluation", []):
#                 st.subheader(f"### {comp.get('company')}")
#                 score = comp.get("relevance_score", 0)
#                 st.progress(min(1, score / 100))
#                 st.markdown(f"**Score:** {score}/100")
#                 # Render per-question assessments if available
#                 qa = comp.get("question_assessments") or []
#                 if qa:
#                     import pandas as _pd
#                     dfq = _pd.DataFrame(qa)
#                     if not dfq.empty:
#                         dfq = dfq[[c for c in ["question", "answer", "score", "justification"] if c in dfq.columns]]
#                         st.markdown("**Primary Questions Assessment:**")
#                         st.dataframe(dfq, use_container_width=True)
#                 if comp.get("fit_summary"):
#                     st.markdown(f"**Fit Summary:** {comp['fit_summary']}")
#                 if comp.get("strengths"):
#                     with st.expander("Strengths", expanded=True):
#                         for s in comp["strengths"]:
#                             st.markdown(f"- {s}")
#                 if comp.get("weaknesses"):
#                     with st.expander("Weaknesses", expanded=False):
#                         for w in comp["weaknesses"]:
#                             st.markdown(f"- {w}")
#                 if comp.get("why_recommended"):
#                     st.markdown(f"**Why Recommended:** {comp['why_recommended']}")
#                 if comp.get("why_not_others"):
#                     st.markdown(f"**Why Not Others:** {comp['why_not_others']}")

#             # Best Recommendation
#             best = result.get("best_recommendation", {}) or {}
#             st.header("🏆 Best Recommended Company")
#             st.markdown(f"**Company:** {best.get('company','N/A')}")
#             st.markdown(f"**Go / No-Go:** {best.get('go_no_go','N/A')}")
#             if best.get("go_no_go_rationale_2_3_lines"):
#                 st.caption(best["go_no_go_rationale_2_3_lines"])
#             if best.get("justification"):
#                 st.markdown(f"**Justification:** {best['justification']}")

#             # JSON summary (single line)
#             def one_line_summary(summary_obj: dict) -> str:
#                 try:
#                     text = (summary_obj.get("scope_of_work") or "").strip()
#                     if not text and summary_obj.get("project_info_points"):
#                         text = "; ".join([str(x) for x in summary_obj.get("project_info_points", []) if str(x).strip()])
#                     if not text and summary_obj.get("key_requirements"):
#                         text = "; ".join([str(x) for x in summary_obj.get("key_requirements", []) if str(x).strip()])
#                     text = " ".join(str(text).split())
#                     return text[:220]
#                 except Exception:
#                     return "N/A"

#             summ = result.get("summary", {}) or {}
#             proj_points = [str(x) for x in (summ.get("project_info_points") or []) if str(x).strip()]
#             key_reqs = [str(x) for x in (summ.get("key_requirements") or []) if str(x).strip()]
#             recs = [str(x) for x in (summ.get("specific_recommendations_before_bidding") or []) if str(x).strip()]

#             summary_4_lines = [
#                 ("Scope: " + (summ.get("scope_of_work") or "N/A")).strip()[:220],
#                 f"Project Info Points (qty {len(proj_points)}): " + "; ".join(proj_points[:5])[:220],
#                 f"Key Requirements (qty {len(key_reqs)}): " + "; ".join(key_reqs[:5])[:220],
#                 f"Recommendations Before Bidding (qty {len(recs)}): " + "; ".join(recs[:5])[:220],
#             ]

#             compact = {
#                 "company": best.get("company", "N/A"),
#                 "go_no_go": best.get("go_no_go", "N/A"),
#                 "scope": one_line_summary(summ),
#                 "summary_4_lines_with_quantity": summary_4_lines,
#             }
#             st.subheader("📦 Compact JSON Result")
#             st.json(compact)

# else:
#     st.caption("⬆️ Please upload an RFP to begin analysis.")




# Streamlit RFP Analyzer (docTR OCR + GPT-4 + .env + Token-safe Top 10 Chunks)
# ----------------------------------------------------------------------------
# Upload RFP + optional company PDFs/DOCX (IKIO, METCO, Sunsprint)
# OCR via docTR, GPT-4 analysis (limited to 10 most relevant text chunks)
# Fully integrated BID Evaluation logic — evaluates every step exactly as in doc file.
# Displays all section names, every step, uses real company names, computes scores,
# and automatically applies state-based bonus scoring using company base locations.
# Extracted RFP details (Step 1) are displayed in tabular format.
# ----------------------------------------------------------------------------

import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from docx import Document

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# Optional PDF export dependency
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --------------- Setup ---------------
st.set_page_config(page_title="RFP Analyzer (GPT-4 + docTR)", layout="wide")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------- Constants ---------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_ROOT, "company_context_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
COMPANY_LOCATIONS = {
    "IKIO": "Indianapolis, IN",
    "Sunsprint Engineering": "Batesville, IN",
    "METCO Engineering": "Dallas, TX"
}

MAPPING_TEXT = """
- Lighting (Supply +Subsitution Allowed) → IKIO
- Lighting (Supply+Installation+Subsitution Not allowed) → Sunsprint Engineering or METCO Engineering
- Lighting (Supply + Installation +Subsitution Allowed) → IKIO or METCO Engineering or Sunsprint Engineering
- HVAC → Sunsprint Engineering or METCO Engineering
- Solar PV → Sunsprint Engineering or METCO Engineering
- Lighting (Installation) → Sunsprint Engineering or METCO Engineering
- Water Management → Sunsprint Engineering or METCO Engineering
- Building Envelope → Sunsprint Engineering or METCO Engineering
- Construction → Sunsprint Engineering or METCO Engineering
- ESCO → Sunsprint Engineering or METCO Engineering
- Emergency Generator → Sunsprint Engineering or METCO Engineering
""".strip()

EVAL_STEPS_TEXT = """
GENERALIZED RFP EVALUATION TABLES FOR EPC COMPANIES
-------------------------------------------------
Step 1: Extract the following information from the RFP.
Step 2: Write a concise summary of the scope of work and key requirements/documents.
Step 3: Answer the following information from the RFP:
  a. What is the project State?
  b. What kind of license is required to work on the project?
  c. Is any site investigation/visit required/mandatory?
  d. Are any specific procurement requirements (BABA/BAA/Davis Bacon) are there?
  e. What specific qualifications are required for the project?
  f. Does the project require SBE, MBE, WBE, HUB goals?
  g. Does the project require and specific security clearance or working hour restrictions?
  h. Is any bond (payment/performance/bid) is required?
  i. Is any insurance is required?

COMPANY SELECTION CRITERIA
--------------------------
Step 1: Recommend applicable companies per mapping.
Step 2: Assign 5 base points to each recommended company.
Step 3: Add 5 bonus points if project state matches company’s state.
Step 4: Suggest best companies accordingly (if all three companies have equal points, then evaluate scoring for all three companies).
Step 5 (Override Rule): If the detected work profile is "Lighting (Supply + Installation +Subsitution Allowed)", you MUST evaluate and score ALL THREE companies (IKIO, METCO Engineering, Sunsprint Engineering) in the BID Evaluation Process, not just top two.

BID EVALUATION PROCESS
----------------------
Step 1: Answer these questions for the best 2 recommended company with their company’s documents and compare with the RFP’s response, then give remarks and score according to the criteria mentioned:

a. Is project state and company state same?  
Scoring Criteria: If the project state and company’s state are same, give score 10, else 0. If 0, then give recommendation.

b. Is the company or its subcontractor (if available) has required license as mentioned in the BID document in the project state (if required/mandatory)?  
Scoring Criteria: If the required license in the BID document in the project state is available with the company or subcontractor, give score 10, else 0. If 0, then give recommendation. If the license required is not mandatory in the BID document then score 10.

c. If site investigation/visit required/mandatory, can company/subcontractor do this?  
Scoring Criteria: If the required site investigation/visit is required/mandatory in the project and company’s or subcontractor are in the same area, give score 10, else 0. If 0, then give recommendation. If the site visit is not required/mandatory, give score 10.

d. Is company capable of fulfilling specific procurement requirements (BABA/BAA/Davis Bacon) (if required/mandatory)?  
Scoring Criteria: If the company is capable of fulfilling specific procurement requirements (BABA/BAA/Davis Bacon) (if required/mandatory), give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

e. Is company capable of fulfilling specific qualifications for the project (if mandatory/required)?  
Scoring Criteria: If the company is capable of fulfilling specific qualifications, give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

f. Can companies meet SBE, MBE, WBE, HUB goals (if required)?  
Scoring Criteria: If the company is capable of fulfilling SBE, MBE, WBE, HUB goals (if required), give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

g. Can company meet specific security clearance or working hour restrictions (if required)?  
Scoring Criteria: If the company is capable of meeting specific security clearance or working hour restrictions (if required), give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

h. Can company provide bond (payment/performance/bid) (if required)?  
Scoring Criteria: If the company is capable of providing bond (payment/performance/bid) (if required), give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

i. Can company provide insurance (if required)?  
Scoring Criteria: If the company is capable of providing insurance (if required), give score 10, else 0. If 0, then give recommendation. If it is not mandatory/required, give score 10.

Step 2: Produce a BID Evaluation Table listing: Question, Score, Remark, Recommendation.
Step 3: Compute total score per company.
Step 4: Determine Go/No-Go per company with rationale.
Step 5: If scores tie, use qualifications fit for tiebreaker.
Step 6: Provide final recommendations to qualify for BID.
""".strip()

# (rest of the script remains unchanged, continuing with OCR extraction, GPT-4 evaluation, and Streamlit UI logic as before)


# --------------- Rule-based Company Recommendation (Step 1 logic) ---------------
# Static mapping from detected project type to recommended companies
PROFILE_TO_COMPANIES = {
    "Lighting (Supply + Substitution Allowed)": ["IKIO"],
    "Lighting (Supply + Installation + Substitution Not allowed)": ["Sunsprint Engineering", "METCO Engineering"],
    "Lighting (Supply + Installation + Substitution Allowed)": ["IKIO", "METCO Engineering", "Sunsprint Engineering"],
    "HVAC": ["Sunsprint Engineering", "METCO Engineering"],
    "Solar PV": ["Sunsprint Engineering", "METCO Engineering"],
    "Lighting (Installation)": ["Sunsprint Engineering", "METCO Engineering"],
    "Water Management": ["Sunsprint Engineering", "METCO Engineering"],
    "Building Envelope": ["Sunsprint Engineering", "METCO Engineering"],
    "Construction": ["Sunsprint Engineering", "METCO Engineering"],
    "ESCO": ["Sunsprint Engineering", "METCO Engineering"],
    "Emergency Generator": ["Sunsprint Engineering", "METCO Engineering"],
}

US_STATE_ABBRS = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
}

def extract_project_state_simple(text: str) -> str | None:
    # Robust 2-letter state extraction: prefer patterns like "City, ST" or "State: ST"
    import re
    states_alt = "|".join(sorted(list(US_STATE_ABBRS)))
    txt = text.upper()
    # Pattern 1: City, ST
    m = re.search(r",\s*(" + states_alt + r")\b", txt)
    if m:
        return m.group(1)
    # Pattern 2: State: ST or State - ST
    m = re.search(r"\bSTATE\s*[:\-]\s*(" + states_alt + r")\b", txt)
    if m:
        return m.group(1)
    return None

def detect_work_profiles(rfp_text: str) -> list[str]:
    t = rfp_text.lower()
    detected: list[str] = []

    if "lighting" in t:
        supply = "supply" in t
        install = ("install" in t) or ("installation" in t) or ("replace" in t)
        # Strict substitution detectors; ensure explicit phrases appear
        no_sub = any(p in t for p in ["no substitution", "no substitutions", "substitution not allowed", "no alternate", "no alternates", "or equal not allowed"])
        yes_sub = any(p in t for p in ["substitution allowed", "alternates allowed", "or equal allowed", "approved equal", "allow substitutions", "allow alternates"])
        if supply and install and no_sub:
            detected.append("Lighting (Supply + Installation + Substitution Not allowed)")
        elif supply and install and yes_sub:
            detected.append("Lighting (Supply + Installation + Substitution Allowed)")
        elif supply and yes_sub:
            detected.append("Lighting (Supply + Substitution Allowed)")
        elif install:
            detected.append("Lighting (Installation)")

    if any(k in t for k in ["hvac", "rtu", "rooftop unit", "air handling", "air handler", "vav", "chiller", "boiler"]):
        detected.append("HVAC")

    if any(k in t for k in ["solar", "photovoltaic", "pv array", "pv system"]):
        detected.append("Solar PV")

    if any(k in t for k in ["plumbing", "water fixture", "flush valve", "urinal", "toilet", "water management"]):
        detected.append("Water Management")

    if any(k in t for k in ["roof", "roofing", "insulation", "fenestration", "window", "door", "envelope"]):
        detected.append("Building Envelope")

    if any(k in t for k in ["general contractor", "renovation", "civil", "structural", "construction work"]):
        detected.append("Construction")

    if "esco" in t:
        detected.append("ESCO")

    if any(k in t for k in ["generator", "genset", "standby power", "emergency generator"]):
        detected.append("Emergency Generator")

    # Deduplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for p in detected:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def compute_recommendations_and_points(rfp_text: str) -> tuple[list[str], dict, list[str], str | None]:
    profiles = detect_work_profiles(rfp_text)
    project_state = extract_project_state_simple(rfp_text)
    # Base points and aggregation
    company_points: dict[str, int] = {"IKIO": 0, "METCO Engineering": 0, "Sunsprint Engineering": 0}
    for profile in profiles:
        for comp in PROFILE_TO_COMPANIES.get(profile, []):
            company_points[comp] = company_points.get(comp, 0) + 5
    # State bonus
    if project_state:
        for comp, loc in COMPANY_LOCATIONS.items():
            try:
                state = loc.split(",")[-1].strip().upper()
                if state == project_state:
                    company_points[comp] = company_points.get(comp, 0) + 5
            except Exception:
                pass
    # Sort companies by points descending
    ordered = sorted(company_points.items(), key=lambda x: (-x[1], x[0]))
    top_companies = [c for c, _ in ordered if _ > 0] or list(company_points.keys())
    return profiles, company_points, top_companies, project_state

def compute_points_table_rows(rfp_text: str):
    profiles = detect_work_profiles(rfp_text)
    # Union of recommended companies across detected profiles
    recommended = set()
    for p in profiles:
        recommended.update(PROFILE_TO_COMPANIES.get(p, []))
    project_state = extract_project_state_simple(rfp_text)
    rows = []
    for comp, loc in COMPANY_LOCATIONS.items():
        comp_state = loc.split(",")[-1].strip().upper()
        # Base points should only be awarded if at least one detected profile maps to the company
        base = 5 if comp in recommended else 0
        bonus = 5 if (project_state and comp_state == project_state) else 0
        rows.append({
            "Company Name": comp,
            "Base Points": base,
            "State Bonus": bonus,
            "Total Points": base + bonus,
            "Justification": f"Base location in {comp_state}, project in {project_state or 'Unknown'}",
        })
    rows = sorted(rows, key=lambda r: (-r["Total Points"], r["Company Name"]))
    allowed = [r["Company Name"] for r in rows if r["Total Points"] > 0] or [r["Company Name"] for r in rows]
    if "Lighting (Supply + Installation + Substitution Allowed)" in profiles:
        allowed = ["IKIO", "METCO Engineering", "Sunsprint Engineering"]
    return profiles, project_state, rows, allowed


# --------------- Company snippets extraction to ground justifications ---------------
def _split_sentences(text: str) -> list[str]:
    # lightweight sentence splitter
    import re
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 3]

CRITERIA_KEYWORDS = {
    "state": ["state", "located", "office", "headquarters", "hq", "registered in"],
    "license": ["license", "licensed", "licensing", "registration", "contractor license"],
    "site_visit": ["site visit", "site investigation", "pre-bid", "walkthrough"],
    "procurement": ["BABA", "BAA", "Davis Bacon", "domestic", "made in"],
    "qualifications": ["qualification", "experience", "capability", "certification"],
    "sbe_mbe": ["SBE", "MBE", "WBE", "HUB", "DBE", "participation"],
    "security": ["clearance", "background", "badging", "working hours", "off-hours"],
    "bond": ["bond", "performance", "payment", "bid bond", "surety"],
    "insurance": ["insurance", "general liability", "workers' compensation", "auto", "umbrella"],
    "hvac_rtu": ["HVAC", "RTU", "rooftop unit", "air handler", "AHU", "mechanical"],
    "permits_codes": ["permit", "permitting", "code", "IBC", "IMC", "NEC"],
    "bas": ["BAS", "building automation", "controls", "integration", "BACnet", "LonWorks"],
    "timeline": ["schedule", "timeline", "lead time", "duration"],
    "budget": ["budget", "cost", "pricing", "estimate"]
}

def build_company_snippets(docs: dict[str, str], max_per_key: int = 6) -> dict[str, dict[str, list[str]]]:
    snippets: dict[str, dict[str, list[str]]] = {}
    for company, text in docs.items():
        sentences = _split_sentences(text[:120000])
        comp_map: dict[str, list[str]] = {k: [] for k in CRITERIA_KEYWORDS.keys()}
        for s in sentences:
            ls = s.lower()
            for key, kws in CRITERIA_KEYWORDS.items():
                if len(comp_map[key]) >= max_per_key:
                    continue
                if any(kw.lower() in ls for kw in kws):
                    comp_map[key].append(s.strip())
        snippets[company] = comp_map
    return snippets


# --------------- Company Cache (append-only) ---------------
def _safe_company_filename(name: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name.strip())
    if not cleaned:
        cleaned = "company"
    return cleaned + ".txt"

def save_company_cache(company_name: str, source_filename: str, extracted_text: str) -> None:
    if not company_name or not extracted_text:
        return
    filepath = os.path.join(CACHE_DIR, _safe_company_filename(company_name))
    try:
        with open(filepath, "a", encoding="utf-8", errors="ignore") as f:
            f.write("\n\n" + ("#" * 20) + "\n")
            f.write(f"Source: {source_filename} | Appended: ")
            import datetime as _dt
            f.write(_dt.datetime.utcnow().isoformat() + "Z\n\n")
            f.write(extracted_text.strip() + "\n")
    except Exception:
        pass

def load_all_company_cache() -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for fname in os.listdir(CACHE_DIR):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(CACHE_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                company = os.path.splitext(fname)[0]
                if content and content.strip():
                    data[company] = content
            except Exception:
                continue
    except Exception:
        pass
    return data

# --------------- OCR Extraction ---------------
@st.cache_data(show_spinner=False)
def extract_text(file_bytes, filename):
    if filename.lower().endswith(".docx") and DocxDocument is not None:
        try:
            docx = DocxDocument(BytesIO(file_bytes))
            return "\n".join([(p.text or "").strip() for p in docx.paragraphs if (p.text or "").strip()])
        except Exception:
            pass

    predictor = ocr_predictor(pretrained=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if filename.lower().endswith(".pdf"):
            doc = DocumentFile.from_pdf(tmp_path)
        else:
            img = Image.open(tmp_path).convert("RGB")
            doc = DocumentFile.from_images([img])
        result = predictor(doc)
        export = result.export()
        text = "\n".join(
            [
                " ".join(w.get("value", "") for w in line.get("words", []))
                for page in export.get("pages", [])
                for block in page.get("blocks", [])
                for line in block.get("lines", [])
            ]
        )
        return text
    finally:
        os.remove(tmp_path)

# --------------- GPT-4 Analysis ---------------
def analyze_with_gpt(rfp_text, company_docs):
    # Reduce chunk size to lower token usage
    chunks = [rfp_text[i:i+1000] for i in range(0, len(rfp_text), 1000)]
    if len(chunks) > 8:
        keywords = ["scope", "work", "bid", "lighting", "hvac", "installation", "requirements", "solar", "construction", "license", "proposal", "project", "compliance", "timeline", "schedule"]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(chunks + [" ".join(keywords)])
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        top_idx = np.argsort(sims)[::-1][:8]
        chunks = [chunks[i] for i in top_idx]
    selected_text = "\n\n".join(chunks)

    # Precompute Step 1 selection and restrict to allowed companies to avoid bloating the prompt
    profiles, project_state, points_rows, allowed_companies = compute_points_table_rows(rfp_text)
    allowed_docs = {k: v for k, v in (company_docs or {}).items() if k in set(allowed_companies)}
    # Build evidence snippets only (omit full documents to stay under token limits)
    company_snippets = build_company_snippets(allowed_docs, max_per_key=4) if allowed_docs else {}

    # Step 1 rule-based recommendation to guide GPT
    profiles, project_state, points_rows, allowed_companies = compute_points_table_rows(rfp_text)
    # Override: for Lighting (Supply + Installation + Substitution Allowed) evaluate all three
    if "Lighting (Supply + Installation + Substitution Allowed)" in profiles:
        allowed_companies = ["IKIO", "METCO Engineering", "Sunsprint Engineering"]

    system_prompt = f"""
You are a senior EPC Bid Evaluation Specialist.
Analyze the given RFP strictly as per the BID Evaluation Steps below.

{EVAL_STEPS_TEXT}

FORMAT & RULES:
- In Step 1, show extracted RFP information as a markdown table with columns:
  | Project Type | Location/Address | State/Region | Owner/Client | Submission Deadline | Pre-Bid/Site Visit Date | Questions Due Date |
  Fill the table with RFP data or 'Not Found'.

- Under COMPANY SELECTION CRITERIA:
  - Use these company base locations for state comparison:
    * IKIO: Indianapolis, IN
    * Sunsprint Engineering: Batesville, IN
    * METCO Engineering: Dallas, TX
  - If project state matches company state, assign +5 bonus points.
  - Display results in a points table with justification for each company. Use the following exact rows computed from the RFP text:
    {points_rows}
  - Compute total scores and list top 2 companies with reasoning. EXCEPTION: If the work profile is "Lighting (Supply + Installation + Substitution Allowed)", list and score ALL THREE companies.

- Under BID EVALUATION PROCESS, for EACH allowed company render ONE markdown table with columns: Question | Score | Remark | Recommendation. The rows MUST use the following wording EXACTLY (verbatim, do not paraphrase):
  a. Is project state and company state same?
  b. Is the company or its subcontractor if available has required license as mentioned in the BID document in the project state (if required/mandatory)?
  c. If site investigation/visit required/mandatory, can company/subcontractor do this?
  d. Is company capable of fulfilling specific procurement requirements (BABA/BAA/Davis Bacon) (if required/mandatory)?
  e. Is company capable of fulfilling specific qualifications for the project (if mandatory/required)?
  f. Can companies meet SBE, MBE, WBE, HUB goals (if required)?
  g. Can company meet specific security clearance or working hour restrictions (if required)?
  h. Can company provide bond (payment/performance/bid) (if required)?
  i. Can company provide insurance (if required)?
  Scoring: strictly follow the criteria in the steps above. In Remark, compare RFP vs the specific company document and quote short evidence (RFP line or company capability line). In Recommendation, write a concise action only when Score < 10; otherwise use “-”. After the table, show Total Score and Go/No-Go with a one-line rationale.
- Always use company names explicitly.
- Do not output JSON.
- Use markdown tables for all scoring data.
 - For STEP 3 above, produce a markdown table named "Compliance and Qualifications — Answers from RFP" with two columns: "Question" and "Answer from RFP (verbatim)". Quote short text directly from the RFP; if not present, write "Not Found".

FINAL RECOMMENDATION SECTION (MANDATORY):
- After the company evaluation tables, compute a context-based percentage for each allowed company.
  Percentage = round(100 * (Total Score / (number_of_questions * 10))).
- Select the best 2 companies by percentage and render:
  1) "Best 2 Companies (with percentage and rationale)" — bullet list with company name, percentage %, and 1–2 line rationale grounded in evidence snippets.
  2) "Recommended Company (Go/No-Go)" — pick the top company with a single sentence justification.
- Then print a section "Generated Summary (Full)" which is the complete generated narrative summary of the RFP and evaluation in 8–12 lines, including scope, key dates, location/state, procurement/qualification requirements, and any critical recommendations before bidding.

PRECOMPUTED STEP 1 GUIDANCE (use exactly as constraints):
- Detected work profiles: {', '.join(profiles) if profiles else 'None detected'}
- Project state (detected): {project_state or 'Unknown'}
- Allowed companies for detailed evaluation: {', '.join(allowed_companies)}
- Restrict detailed BID EVALUATION PROCESS tables and Go/No-Go to only the allowed companies above (except when showing the Step 1 points table itself).

CRITICAL INSTRUCTION FOR RECOMMENDATIONS COLUMN:
- Derive each Recommendation directly from the corresponding company's cached profile evidence provided below (EVIDENCE SNIPPETS). Do not guess.
- If evidence shows capability is present → Recommendation must be "-".
- If evidence is partial or absent → write a short, actionable fix, e.g.:
  • For license: "Engage TX-licensed subcontractor" or "Obtain TX mechanical contractor license before bid".
  • For site visit: "Schedule pre-bid/site visit; assign local team".
  • For procurement (BABA/BAA/Davis Bacon): "Source compliant materials; attach vendor/compliance plan".
  • For qualifications: "Attach similar project references meeting min quals".
  • For SBE/MBE/WBE/HUB: "Partner with certified local firm; include commitment letter".
  • For security/hours: "Confirm background checks; plan off-hours work if needed".
  • For bonds/insurance: "Confirm surety line and COI limits; secure bid/performance/payment bonds".
  • For BAS: "Coordinate with existing BAS vendor; confirm BACnet points list".
  • For permits/codes: "Prepare permit set; confirm local code compliance checklist".
  • For timeline/budget: "Adjust schedule/resources; value-engineer per RFP budget".
  Always keep Recommendation ≤ 12 words and only include when Score < 10; otherwise "-".
""".strip()

    user_prompt = f"RFP EXTRACTED CONTENT:\n{selected_text}\n\n"
    user_prompt += "PRECOMPUTED STEP 1 CONTEXT (for your reference):\n"
    user_prompt += f"Detected Profiles: {', '.join(profiles) if profiles else 'None'}\n"
    user_prompt += f"Project State: {project_state or 'Unknown'}\n"
    user_prompt += f"Allowed Companies: {', '.join(allowed_companies)}\n\n"
    if company_snippets:
        user_prompt += "EVIDENCE SNIPPETS (use for justification; quote when relevant):\n"
        for cname, cmap in company_snippets.items():
            user_prompt += f"Company: {cname}\n"
            for key, sents in cmap.items():
                if sents:
                    joined = " | ".join(sents[:4])
                    user_prompt += f"- {key}: {joined}\n"
            user_prompt += "\n"

    try:
        response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                top_p=1,
                seed=42,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback for token/TPM errors: shrink context and retry once
        if "rate_limit_exceeded" in str(e).lower() or "request too large" in str(e).lower():
            small_selected = "\n\n".join(chunks[:5])
            # reduce snippets per key
            small_snips = build_company_snippets(allowed_docs, max_per_key=2) if allowed_docs else {}
            small_prompt = f"RFP EXTRACTED CONTENT (condensed):\n{small_selected}\n\n"
            small_prompt += f"Allowed Companies: {', '.join(allowed_companies)}\n\n"
            if small_snips:
                small_prompt += "EVIDENCE SNIPPETS (condensed):\n"
                for cname, cmap in small_snips.items():
                    small_prompt += f"Company: {cname}\n"
                    for key, sents in cmap.items():
                        if sents:
                            joined = " | ".join(sents[:2])
                            small_prompt += f"- {key}: {joined}\n"
                    small_prompt += "\n"
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                top_p=1,
                seed=42,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": small_prompt},
                ],
            )
            return response.choices[0].message.content
        raise

# --------------- Word Download Helper ---------------
def create_docx_from_text(text):
    doc = Document()
    for line in text.split("\n"):
        if line.strip() == "":
            doc.add_paragraph("")
        elif line.startswith("#"):
            doc.add_heading(line.replace("#", "").strip(), level=1)
        elif line.lower().startswith("step"):
            doc.add_heading(line.strip(), level=2)
        else:
            doc.add_paragraph(line)
    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output

# --------------- Streamlit UI ---------------
st.title("📄 RFP Analyzer — BID Evaluation with Tables, Scoring & State Logic")

rfp_file = st.file_uploader("📤 Upload RFP/BID Document", type=["pdf", "png", "jpg", "jpeg", "tiff", "docx"])

st.subheader("🏢 Upload Company Profiles (optional)")
c1, c2, c3 = st.columns(3)
with c1:
    ikio_file = st.file_uploader("IKIO", type=["pdf", "jpg", "jpeg", "png", "docx"], key="ikio")
with c2:
    metco_file = st.file_uploader("METCO Engineering", type=["pdf", "jpg", "jpeg", "png", "docx"], key="metco")
with c3:
    suns_file = st.file_uploader("Sunsprint Engineering", type=["pdf", "jpg", "jpeg", "png", "docx"], key="suns")

# General uploader to add any future company; cache is append-only
with st.expander("➕ Add another company (persistent cache)"):
    new_company_name = st.text_input("Company Name (exact display name)")
    new_company_file = st.file_uploader("Upload Company Document", type=["pdf", "jpg", "jpeg", "png", "docx"], key="new_company")
    if new_company_name and new_company_file is not None:
        st.info("Reading uploaded company document…")
        _txt = extract_text(new_company_file.read(), new_company_file.name)
        if _txt and _txt.strip():
            save_company_cache(new_company_name, new_company_file.name, _txt)
            st.success(f"Cached content appended for {new_company_name}.")
        else:
            st.warning("No text extracted from the uploaded file.")

if rfp_file:
    st.info("Extracting RFP text…")
    rfp_text = extract_text(rfp_file.read(), rfp_file.name)
    if not rfp_text.strip():
        st.error("No text extracted.")
        st.stop()
    st.success("RFP text ready.")

    # Start from cached profiles
    company_docs = load_all_company_cache()
    # Merge in any newly uploaded defaults
    if ikio_file:
        _t = extract_text(ikio_file.read(), ikio_file.name)
        company_docs["IKIO"] = (company_docs.get("IKIO", "") + "\n" + _t).strip()
        save_company_cache("IKIO", ikio_file.name, _t)
    if metco_file:
        _t = extract_text(metco_file.read(), metco_file.name)
        company_docs["METCO Engineering"] = (company_docs.get("METCO Engineering", "") + "\n" + _t).strip()
        save_company_cache("METCO Engineering", metco_file.name, _t)
    if suns_file:
        _t = extract_text(suns_file.read(), suns_file.name)
        company_docs["Sunsprint Engineering"] = (company_docs.get("Sunsprint Engineering", "") + "\n" + _t).strip()
        save_company_cache("Sunsprint Engineering", suns_file.name, _t)

    if st.button("🚀 Run Evaluation (Apply State Match Scoring)"):
        with st.spinner("Evaluating with location-based bonus scoring and tables…"):
            result = analyze_with_gpt(rfp_text, company_docs)
        st.markdown(result, unsafe_allow_html=True)

        if REPORTLAB_AVAILABLE:
            def create_pdf_from_text(text):
                output = BytesIO()
                doc = SimpleDocTemplate(output, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 12))
                    else:
                        story.append(Paragraph(line, styles["Normal"]))
                doc.build(story)
                output.seek(0)
                return output

            st.download_button(
                label="💾 Download Evaluation as PDF File",
                data=create_pdf_from_text(result),
                file_name="RFP_BID_Evaluation_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("reportlab is not installed; providing DOCX download instead.")
            st.download_button(
                label="💾 Download Evaluation as Word File",
                data=create_docx_from_text(result),
                file_name="RFP_BID_Evaluation_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

else:
    st.caption("⬆️ Please upload an RFP to start.")
