SCOPE_CLASSIFIER_PROMPT = """
You are classifying user questions for a research synthesis system.

The system ONLY covers peer-reviewed academic research on the SOCIAL dimensions of renewable energy adoption or energy transition.

This includes research on:

1) Social impacts and outcomes, such as effects on:
- people, households, communities, or social groups
- costs, affordability, economic well-being
- distributional effects, equity, or energy poverty
- governance, institutions, or decision-making processes

2) Social processes shaping renewable energy adoption or energy transition, such as:
- public opinion, attitudes, or perception
- social acceptance or opposition
- education, awareness, or energy literacy
- participation, engagement, or community involvement
- trust, legitimacy, or social norms

The system does NOT cover:
- physical climate effects
- environmental or ecological impacts
- emissions, temperature, or climate system changes
- purely technical, engineering, or performance issues
- purely natural science questions without a social dimension

If the question primarily concerns social dimensions of renewable energy adoption or energy transition, answer "yes".

If it primarily concerns environmental, technical, engineering, or climate system issues without a social focus, answer "no".

Answer ONLY with "yes" or "no".

Question:
{{QUESTION}}
"""


RELEVANCE_JUDGE_PROMPT = """
You are evaluating the relevance of a set of passages for answering a user query.

TASK:
Assign a relevance score to each passage independently, based on how useful it is for answering the query.

SCORING RULE:

1 = Relevant: the passage contains VERY VALUABLE information to answer the query.
0 = Not relevant: the information in the passage is indirect, too general, or only weakly useful.

IMPORTANT RULES:
- BE STRICT
- Do NOT mark a passage as relevant just because it shares keywords.
- Focus on usefulness for answering the query, not just topic similarity.
- If unsure, choose 0.
- Do NOT try to distribute scores evenly. Each passage must be evaluated independently.
- Do NOT assume earlier passages are more relevant.

OUTPUT FORMAT (STRICT):
Return ONLY a JSON list of integers (0 or 1), one per passage:
[is_relevant(query, passage_1) = 0|1, ..., is_relevant(query, passage_N) = 0|1]

NO explanation. NO extra text.

REMINDER:
EVALUATE EACH PASSAGE AGAINST THE QUERY INDEPENDENTLY FROM THE OTHER PASSAGES.
---

QUERY:
{{query}}

---

PASSAGES:
{{passages}}

"""


# QUERY_EXPANDER_PROMPT = """
# You reformulate research questions to improve retrieval in academic databases. 

# Generate 3 alternative versions of the query that preserve the original meaning and scope. 

# Each reformulation MUST: 
# - Use different academic terminology or phrasing 
# - Maintain the same level of specificity (do NOT broaden or narrow the topic) 
# - Preserve the original intent exactly 

# Ensure diversity across the reformulations: 
# - One should favor formal academic phrasing 
# - One should favor common terminology used in literature 
# - One may vary key terms (e.g., synonyms such as "impact", "relationship", "effect") 

# Do NOT introduce new concepts, sub-questions, or assumptions. 
# Preserve key domain terms when possible (e.g., "renewable energy", "solar energy"). 

# OUTPUT FORMAT (IMPORTANT): 
# You MUST return the query reformulations as a list, using this format: 
# ["expanded_query_1", "expanded_query_2", ...] 

# Original query: 
# {{QUESTION}}
# """

# QUERY_EXPANDER_PROMPT = """
# You reformulate research questions to improve retrieval in academic databases.

# Generate 3 alternative versions of the query that preserve the original meaning and scope.

# Requirements:
# - Keep each reformulation concise and focused
# - Preserve the original intent exactly (no added or removed concepts)
# - Use different wording, phrasing, or structure to increase retrieval diversity
# - Vary important terms where appropriate (e.g., synonyms), but do NOT distort meaning
# - Avoid repeating the same phrasing patterns across outputs

# Do NOT:
# - Introduce new topics, sub-questions, or assumptions
# - Overly lengthen or complicate the query
# - Drift away from the original focus

# OUTPUT FORMAT (STRICT):
# Return a JSON list of strings:
# ["query1", "query2", "query3"]

# Original query:
# {{QUESTION}}
# """

QUERY_EXPANDER_PROMPT = """
You reformulate research questions to improve retrieval in academic databases.

Generate 3 alternative versions of the query.

Requirements:
- Preserve the core meaning of the original query
- Do NOT introduce unrelated concepts
- Slightly vary terminology and phrasing to improve retrieval diversity
- You may use synonyms or closely related academic expressions
- Keep queries concise and focused

Goal:
Produce variations that could retrieve complementary relevant evidence.

OUTPUT FORMAT:
["query1", "query2", "query3"]

Original query:
{{QUESTION}}
"""

TASK_HEADER = """
You are an expert research assistant specialized in environmental and social impact analysis.
You synthesize evidence strictly from the provided peer-reviewed academic sources.
"""

CORE_SYNTHESIS_INSTRUCTIONS = """
STRICT RULES:
  - Use ONLY the provided sources.
  - Do NOT use external knowledge.
  - Do NOT speculate, generalize, extrapolate, or "fill in gaps".
  - Do NOT infer causal or conceptual links that are not explicitly supported.
  - Use cautious, academic language (e.g., "suggests", "is associated with", "was observed").
  - Do NOT imply global or universal effects unless explicitly stated in the sources.
  - Do NOT prioritize answering over correctness.

MANDATORY DECISION STEP (CRITICAL):
  Before generating the answer, you MUST internally decide:

    SUPPORTED = YES or NO

    SUPPORTED = NO if ANY of the following holds:
    - Sources are unrelated or only loosely related to the question.
    - Sources discuss a different population, technology, outcome, or context than the question requires.
    - Only partial or indirect evidence exists (requires interpretation or bridging).
    - Evidence is too vague, high-level, or generic to support a concrete claim.

    SUPPORTED = YES only if:
    - At least one claim can be directly and explicitly supported by the sources without inference.

    If SUPPORTED = NO:
    - You MUST return:
      {
        "answer": [],
        "limitations": ["Explanation of why the evidence is insufficient or mismatched."]
      }
    - DO NOT attempt to generate any claims.

EVIDENCE REQUIREMENTS:
  - EVERY factual claim MUST be supported by one or more citations.
  - Citations MUST be attached at the sentence level.
  - Each citation MUST be a valid chunk_id from the provided context (e.g., "paper_12__chunk_46").
  - A sentence MAY cite multiple chunks if supported by multiple studies.
  - If a claim cannot be directly supported by at least one chunk, it MUST NOT be included.

CLAIM TYPES (IMPORTANT):
  Each sentence in the answer MUST be one of the following:

    1. Contextual finding:
      - Reports a specific result from an individual study.
      - Typically includes study context (e.g., country, region, population, policy setting, or method).
      - May be supported by a single citation.
      - Numerical or monetary estimates SHOULD be included when reported.

    2. Cross-study pattern:
      - Synthesizes a shared finding observed across multiple independent studies.
      - MUST be supported by citations from more than one paper.
      - MUST NOT merge incompatible study designs, outcomes, or contexts.

  Prefer cross-study patterns when multiple independent sources support the same high-level finding.
  Do NOT force aggregation for highly context-specific or numerical results.

SOURCE BALANCE:
  - Prefer distributing evidence across independent studies when available.
  - Do NOT force inclusion of weakly relevant sources.
  - Do NOT exclude relevant sources unnecessarily.

CRITICAL CONSTRAINTS:
  - Do NOT include citations inside the sentence text.
  - Do NOT mention studies, authors, years, or evidence that are not listed in the "citations" field.
  - The "citations" field is the ONLY place where evidence may be referenced.

TAG GUIDANCE:
  Tags are metadata only and NOT evidence.
  - MUST NOT be cited or used to justify claims.
  - May ONLY be used to help identify coverage of themes.

OUTPUT FORMAT:
  You MUST return ONLY valid JSON.
  Do NOT include explanations, markdown, or additional text.

    JSON SCHEMA:
    {
      "answer": [
        {
          "text": "Single, evidence-based factual claim.",
          "citations": ["chunk_id", "..."]
        }
      ],
      "limitations": [
        "Brief description of uncertainty, scope limitations, or missing evidence."
      ]
    }

FAILURE CONDITIONS (MANDATORY):
  - If SUPPORTED = NO:
    - "answer" MUST be empty.
    - "limitations" MUST clearly explain topic mismatch, OR insufficient evidence, OR lack of direct support.

LIMITATIONS REQUIREMENT (MANDATORY):
  - If "answer" is non-empty:
    - "limitations" MUST contain at least one item.
    - Each item MUST describe uncertainty, scope limits, or evidence gaps.
  - Outputs where "answer" is non-empty and "limitations" is empty are INVALID.

ADDITIONAL INSTRUCTIONS:
  - Sentences must be concise and atomic.
  - Prefer cross-study patterns when strongly supported.

SOURCES:
{{SOURCES}}

QUESTION:
{{QUESTION}}
"""

RETRY_SOURCE_DIVERSITY = """
RETRY INSTRUCTION:
The previous synthesis relied predominantly on one source, despite multiple strong and relevant papers being available.
Re-examine the retrieved evidence and ensure that relevant findings from independent papers are incorporated where they meaningfully contribute to the answer.
Do not introduce unsupported claims or include sources that do not add substantive information.
"""

RETRY_EVIDENCE_UTILIZATION = """
RETRY INSTRUCTION:
The previous synthesis incorporated only a small portion of the retrieved evidence.
Review the provided excerpts and ensure that all substantively relevant and non-redundant findings are considered in the answer.
Prioritize clarity and conciseness; do not repeat similar findings across sentences.
"""

RETRY_CORROBORATION = """
RETRY INSTRUCTION:
The previous synthesis cited multiple papers but did not integrate their findings within individual claims.
Where the evidence allows, synthesize overlapping or convergent findings from independent papers into unified statements.
If a claim is supported by only one source, clearly qualify it rather than overstating its generality.
Do not introduce new claims without evidence.
"""

RETRY_PROMPTS = {
    "source_dominance": RETRY_SOURCE_DIVERSITY,
    "no_corroboration": RETRY_CORROBORATION,
    "low_evidence_usage": RETRY_EVIDENCE_UTILIZATION,
}