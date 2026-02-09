"""
Synthetic Question-Answer generation for Community Reports.

Generates synthetic questions that a community report can answer,
improving semantic retrieval by creating additional indexable content.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# Prompt for generating synthetic questions
SYNTHETIC_QA_PROMPT = """Given the following community report, generate 3-5 specific, diverse questions that this report can answer well.

COMMUNITY REPORT:
Title: {title}
Summary: {summary}
Full Content: {content}

Generate questions that:
1. Are specific and answerable from this report
2. Cover different aspects of the report (not all about the same thing)
3. Use natural language that users might actually ask
4. Vary in style (some broad, some specific, some relational)
5. Are in the same language as the report

Format your response as a numbered list:
Q1: [first question]
Q2: [second question]
Q3: [third question]
Q4: [fourth question] (optional)
Q5: [fifth question] (optional)

Only output the numbered questions, nothing else."""


async def generate_synthetic_questions(
    report: Dict[str, Any],
    llm: Any,
    num_questions: int = 5
) -> List[str]:
    """
    Generate synthetic questions for a community report.

    Args:
        report: Community report dictionary with 'title', 'summary', 'full_report'
        llm: LLM instance for generation
        num_questions: Number of questions to generate (3-5 recommended)

    Returns:
        List of generated questions
    """
    try:
        title = report.get("title", "")
        summary = report.get("summary", "")
        content = report.get("full_report", "")

        # Truncate content if too long (to avoid token limits)
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        prompt = SYNTHETIC_QA_PROMPT.format(
            title=title,
            summary=summary,
            content=content
        )

        # Generate questions
        if hasattr(llm, 'ainvoke'):
            response = await llm.ainvoke(prompt)
            content_response = response.content if hasattr(response, 'content') else str(response)
        elif hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
            content_response = response.content if hasattr(response, 'content') else str(response)
        else:
            raise AttributeError("LLM does not have ainvoke or invoke method")

        # Parse questions from response
        questions = parse_questions_from_text(content_response)

        if not questions:
            logger.warning(f"No questions generated for report: {title}")
            # Generate fallback questions based on title/summary
            questions = generate_fallback_questions(title, summary)

        logger.info(f"Generated {len(questions)} synthetic questions for report: {title}")
        return questions[:num_questions]

    except Exception as e:
        logger.error(f"Error generating synthetic questions: {e}")
        # Return fallback questions
        return generate_fallback_questions(
            report.get("title", ""),
            report.get("summary", "")
        )


def parse_questions_from_text(text: str) -> List[str]:
    """
    Parse questions from LLM response text.

    Looks for patterns like:
    - Q1: question
    - 1. question
    - 1) question

    Args:
        text: Text containing questions

    Returns:
        List of extracted questions
    """
    questions = []

    # Pattern 1: Q1: question
    pattern1 = r'Q\d+:\s*(.+?)(?=Q\d+:|$)'
    matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
    questions.extend([q.strip() for q in matches1 if q.strip()])

    # Pattern 2: 1. question or 1) question
    if not questions:
        pattern2 = r'\d+[\.)]\s*(.+?)(?=\d+[\.)]|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        questions.extend([q.strip() for q in matches2 if q.strip()])

    # Pattern 3: Lines starting with dash or bullet
    if not questions:
        pattern3 = r'[-•]\s*(.+?)(?=[-•]|$)'
        matches3 = re.findall(pattern3, text, re.DOTALL)
        questions.extend([q.strip() for q in matches3 if q.strip()])

    # Clean up questions
    cleaned_questions = []
    for q in questions:
        # Remove newlines and extra spaces
        q = ' '.join(q.split())
        # Remove trailing punctuation if not a question mark
        q = q.rstrip('.,;:')
        # Ensure it ends with a question mark if it looks like a question
        if q and not q.endswith('?'):
            q += '?'
        if q:
            cleaned_questions.append(q)

    return cleaned_questions


def generate_fallback_questions(title: str, summary: str) -> List[str]:
    """
    Generate simple fallback questions when LLM generation fails.

    Args:
        title: Report title
        summary: Report summary

    Returns:
        List of fallback questions
    """
    questions = []

    if title:
        # Extract main subject from title
        subject = title.replace("Community Report:", "").strip()
        questions.append(f"What is {subject}?")
        questions.append(f"Tell me about {subject}.")
        questions.append(f"Can you explain {subject}?")

    if summary:
        # Try to extract key concepts from summary (first few words)
        words = summary.split()[:10]
        key_phrase = ' '.join(words)
        questions.append(f"What does this community focus on: {key_phrase}?")

    # Generic questions
    questions.append("What are the main topics covered in this community?")

    return questions[:5]


async def enrich_reports_with_synthetic_qa(
    reports: List[Dict[str, Any]],
    llm: Any,
    num_questions_per_report: int = 3
) -> List[Dict[str, Any]]:
    """
    Enrich multiple reports with synthetic questions.

    Args:
        reports: List of community report dictionaries
        llm: LLM instance for generation
        num_questions_per_report: Number of questions per report

    Returns:
        List of reports with added 'synthetic_questions' field
    """
    enriched_reports = []

    for idx, report in enumerate(reports):
        try:
            logger.info(f"Generating synthetic questions for report {idx+1}/{len(reports)}")

            questions = await generate_synthetic_questions(
                report=report,
                llm=llm,
                num_questions=num_questions_per_report
            )

            enriched_report = report.copy()
            enriched_report['synthetic_questions'] = questions

            enriched_reports.append(enriched_report)

        except Exception as e:
            logger.error(f"Error enriching report {idx}: {e}")
            # Add report without questions
            enriched_report = report.copy()
            enriched_report['synthetic_questions'] = []
            enriched_reports.append(enriched_report)

    logger.info(f"Enriched {len(enriched_reports)} reports with synthetic questions")
    return enriched_reports


def format_report_with_questions_for_indexing(report: Dict[str, Any]) -> str:
    """
    Format a report with its synthetic questions for vector indexing.

    The questions are prepended to improve semantic retrieval.

    Args:
        report: Report dictionary with 'synthetic_questions' field

    Returns:
        Formatted text for indexing
    """
    title = report.get("title", "")
    summary = report.get("summary", "")
    questions = report.get("synthetic_questions", [])

    parts = []

    # Add questions first (they help with retrieval)
    if questions:
        parts.append("Questions this report answers:")
        for i, q in enumerate(questions, 1):
            parts.append(f"{i}. {q}")
        parts.append("")  # Empty line

    # Add title and summary
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")

    return "\n".join(parts)
