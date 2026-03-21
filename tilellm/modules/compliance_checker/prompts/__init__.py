"""
Built-in domain prompts for ComplianceChecker.

Usage:
    from tilellm.modules.compliance_checker.prompts import get_builtin_config
    config = get_builtin_config("e_procurement")
"""
from tilellm.modules.compliance_checker.prompts.e_procurement import E_PROCUREMENT_CONFIG
from tilellm.modules.compliance_checker.prompts.hr_assessment import HR_ASSESSMENT_CONFIG
from tilellm.modules.compliance_checker.prompts.legal_audit import LEGAL_AUDIT_CONFIG
from tilellm.modules.compliance_checker.prompts.medical_devices import MEDICAL_DEVICES_CONFIG

_REGISTRY = {
    "e_procurement": E_PROCUREMENT_CONFIG,
    "hr_assessment": HR_ASSESSMENT_CONFIG,
    "legal_audit": LEGAL_AUDIT_CONFIG,
    "medical_devices": MEDICAL_DEVICES_CONFIG,
}


def get_builtin_config(domain: str):
    """Return a built-in ComplianceConfig for *domain*, or None if not found."""
    return _REGISTRY.get(domain)


def list_builtin_domains():
    """Return all registered built-in domain names."""
    return list(_REGISTRY.keys())
