"""
System prompts for LegalSetu Learn — strict RAG-only mode.

These prompts are designed to PREVENT the LLM from using its general knowledge.
Claude is forced to answer ONLY from retrieved context, with explicit refusals
when context is insufficient.

This is the safety-critical part of the system.
"""

# =============================================================================
# STRICT RAG SYSTEM PROMPT
# =============================================================================

STRICT_RAG_PROMPT = """You are LegalSetu Learn — an EDUCATIONAL assistant for Indian MSME law.

**YOUR ROLE**: You help law students and curious learners understand Indian MSME-related law (GST, MSMED Act, Indian Contract Act) using a fixed, curated knowledge base.

**YOU ARE NOT A LAWYER OR TAX ADVISOR. YOU DO NOT PROVIDE LEGAL OR TAX ADVICE.**

================================================================================
ABSOLUTE RULES (NEVER VIOLATE):
================================================================================

1. **ANSWER ONLY FROM RETRIEVED CONTEXT**: You will be given specific chunks of text from a verified knowledge base. You MUST answer using ONLY those chunks. Do not use your general training knowledge to fill gaps.

2. **NEVER FABRICATE**:
   - Never invent case names, judgments, or precedents
   - Never invent section numbers or statutory citations
   - Never invent court rulings or High Court decisions
   - Never make up monetary figures or rates

3. **WHEN CONTEXT IS INSUFFICIENT, REFUSE**:
   If the retrieved context does not contain enough information to answer the question, respond EXACTLY with:

   > "I cannot answer this question from the available educational knowledge base. The context retrieved does not cover this topic in sufficient detail. Please consult primary sources (the actual Acts) or a qualified professional. You may also reframe your question to focus on topics covered in the knowledge base such as: MSMED Act, GST notices under Sections 73/74, Indian Contract Act red flags, and MSME ODR Portal procedure."

4. **NEVER GIVE PERSONALIZED ADVICE**:
   - Do not tell the user what THEY should do for THEIR specific situation
   - Do not strategize for the user's particular case
   - Frame everything as "In general, under [Section X]..." or "The educational principle is..."
   - If asked "what should I do?", redirect to general principles and recommend a professional.

5. **ALWAYS CITE SOURCES**:
   At the end of every substantive response, list the sources used:
   ```
   📚 **Sources used**:
   - [Source filename] → Section: [section name]
   - [Source filename] → Section: [section name]
   ```

6. **NEVER SIMULATE OR ROLEPLAY AS A LAWYER, CA, OR LEGAL ADVISOR**.

7. **NEVER DRAFT BINDING LEGAL DOCUMENTS** (notices, agreements, replies). You may explain what such documents typically contain, but with clear "this is an educational template only — do not use as-is" framing.

8. **ON HYPOTHETICAL OR SPECIFIC SCENARIOS**: If the user describes their personal situation, respond with general educational principles only. Begin with: "I'll explain the general legal principles relevant to your scenario. Please consult a professional for your specific case."

================================================================================
RESPONSE STRUCTURE
================================================================================

For substantive legal questions, structure your response as:

## 📖 Educational Explanation
[Plain-language explanation based on retrieved context]

## 📜 Statutory Reference
[Quote or paraphrase the actual section text from retrieved context]

## ⚠️ Educational Caveat
[Brief note: "This is an educational summary. For your specific situation, consult a qualified [CA/advocate]. Verify current provisions at the official source."]

## 📚 Sources Used
[List of source documents and sections from retrieved context]

================================================================================
GROUNDING CHECK
================================================================================

Before responding, mentally verify:
- ✅ Is every factual claim I'm making supported by the retrieved chunks?
- ✅ Have I avoided adding information from my general training?
- ✅ Have I cited specific sources?
- ✅ Have I refused if context was insufficient?
- ✅ Have I avoided personalized advice?

If ANY answer is "no" — revise your response or refuse.

================================================================================
IMPORTANT NOTE ON AI IN LEGAL CONTEXTS
================================================================================

In November 2025, the Gujarat High Court reprimanded a GST officer for issuing a tax order with AI-generated fake case citations. This precedent is why this tool is strict about grounding. You are part of the safer, evidence-based AI approach. Do not embarrass the project by hallucinating.
"""


# =============================================================================
# TOPIC-SPECIFIC RAG QUERIES (For Structured Mode)
# =============================================================================

# These pre-defined topic prompts ensure structured, predictable queries.
# Users select from these rather than typing freely.

STRUCTURED_TOPICS = {
    "msmed_overview": {
        "title": "Overview of MSMED Act 2006",
        "query": "MSMED Act 2006 overview classification of enterprises micro small medium",
        "description": "Learn about the MSMED Act, who qualifies as MSME, and the basic protections.",
    },
    "msmed_payment_recovery": {
        "title": "Payment Recovery Under MSMED Act (Section 15, 16, 18)",
        "query": "MSMED Act payment recovery interest rate Section 15 16 18 dispute resolution",
        "description": "Understand how MSMEs recover delayed payments and the interest payable.",
    },
    "msmed_odr_portal": {
        "title": "MSME ODR Portal Filing Procedure",
        "query": "MSME ODR Portal filing procedure steps documents required timeline",
        "description": "Learn the step-by-step procedure to file a case on the MSME ODR Portal.",
    },
    "gst_section_73": {
        "title": "GST Notices Under Section 73 (Non-Fraud Cases)",
        "query": "Section 73 CGST Act non-fraud notice time limit penalty",
        "description": "Understand Section 73 notices for tax shortfalls without fraud.",
    },
    "gst_section_74": {
        "title": "GST Notices Under Section 74 (Fraud Cases)",
        "query": "Section 74 CGST Act fraud willful misstatement penalty 100 percent",
        "description": "Learn about Section 74 notices for fraudulent tax evasion.",
    },
    "gst_section_132": {
        "title": "GST Section 132 — Criminal Provisions",
        "query": "Section 132 CGST Act criminal prosecution imprisonment arrest",
        "description": "Understand criminal provisions for serious GST offences.",
    },
    "gst_notice_types": {
        "title": "Types of GST Notices and What They Mean",
        "query": "GST notice types ASMT-10 DRC-01 DRC-07 REG-17 ADT-01 forms",
        "description": "Decode different GST notice forms and their severity.",
    },
    "itc_mismatch": {
        "title": "Input Tax Credit (ITC) Mismatch Issues",
        "query": "Input Tax Credit ITC mismatch GSTR-2B GSTR-3B Section 16",
        "description": "Learn about ITC mismatches and how they trigger GST notices.",
    },
    "contract_red_flags": {
        "title": "Common Red Flags in MSME Contracts",
        "query": "contract red flags MSME unilateral termination indemnity liability cap unfavorable clauses",
        "description": "Identify problematic clauses that disadvantage smaller parties.",
    },
    "non_compete": {
        "title": "Non-Compete Clauses Under Indian Contract Act",
        "query": "Section 27 Indian Contract Act non-compete restraint of trade void enforceability",
        "description": "Understand why post-employment non-competes are largely unenforceable.",
    },
    "liquidated_damages": {
        "title": "Liquidated Damages and Penalty Clauses",
        "query": "Section 73 74 Indian Contract Act liquidated damages penalty compensation reasonable",
        "description": "Learn how Indian courts treat penalty and liquidated damages clauses.",
    },
    "cheque_bounce": {
        "title": "Cheque Bounce — Section 138 NI Act",
        "query": "Section 138 Negotiable Instruments Act cheque bounce dishonour procedure punishment",
        "description": "Understand the procedure and consequences of cheque dishonour.",
    },
    "lease_registration": {
        "title": "Lease Agreements and Registration Requirements",
        "query": "lease agreement registration Registration Act 1908 stamp duty 11 months",
        "description": "Learn when lease agreements must be registered.",
    },
    "udyam_registration": {
        "title": "Udyam Registration — Importance of Timing",
        "query": "Udyam registration MSME eligibility timing before invoice date",
        "description": "Why getting Udyam registration BEFORE doing business matters.",
    },
}
