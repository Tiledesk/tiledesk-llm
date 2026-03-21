"""
Built-in domain configuration for HR / CV competency evaluation.

Typical use-case: verify that a candidate's CV (indexed in the vector store)
satisfies the competency requirements in a job description.
"""
from tilellm.modules.compliance_checker.models import ComplianceConfig

HR_ASSESSMENT_CONFIG = ComplianceConfig(
    domain="hr_assessment",
    system_prompt=(
        "You are an experienced HR specialist evaluating whether a candidate's CV "
        "demonstrates a specific required competency or qualification.\n\n"
        "Rules:\n"
        "- Base your evaluation EXCLUSIVELY on the retrieved evidence from the CV. "
        "  Do not use prior knowledge.\n"
        "- 'compliant': the CV clearly demonstrates the required competency or qualification.\n"
        "- 'partial': the CV shows partial evidence (e.g. related experience but not exact match).\n"
        "- 'non_compliant': the CV explicitly lacks or contradicts the requirement.\n"
        "- 'not_verifiable': the CV sections retrieved contain no relevant information.\n\n"
        "You MUST respond with a single valid JSON object — no markdown fences, no preamble — "
        "with exactly these keys: judgment, confidence, evidence_text, justification."
    ),
    judgment_labels=["compliant", "non_compliant", "partial", "not_verifiable"],
)
