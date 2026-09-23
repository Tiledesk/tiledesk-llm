"""
AgenticComplianceChecker — agentic tool exposure for the compliance_checker module.

Wraps compliance_checker's existing services (DiscretionaryCheckService,
resolve_proportional, l01_service, ...) as LangChain tools + an MCP server,
backed by a Redis-persisted session (see docs/README.md and docs/AUDIT.md).
No compliance logic — guardrails, scoring, HyDE fallback — is reimplemented
here; this module only decides *when* to call the existing code and records
*that* it did.

Register this module by placing it under tilellm/modules/; the feature-router
auto-loader in __main__.py will discover controllers.py and mount the router.
"""
