"""ComplianceChecker v2 — service layer."""
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import YamlRequirementsLoader
from tilellm.modules.compliance_checker.services.xlsx_extraction_service import (
    ExtractedLot,
    XlsxExtractionService,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import (
    DiscretionaryCheckService,
    check_compliance_v2,
)
from tilellm.modules.compliance_checker.services.bulk_check_service import (
    check_compliance_v2_bulk,
    resolve_proportional,
)
from tilellm.modules.compliance_checker.services.requirements_xlsx_service import (
    RequirementsXlsxService,
    select_lot,
)
from tilellm.modules.compliance_checker.services.restituzione_xlsx_service import (
    RestituzioneXlsxService,
)
from tilellm.modules.compliance_checker.services.aggregate_report_service import (
    AggregateReportService,
    build_aggregate_report,
    render_report,
)

__all__ = [
    "YamlRequirementsLoader",
    "XlsxExtractionService",
    "ExtractedLot",
    "RequirementsXlsxService",
    "RestituzioneXlsxService",
    "select_lot",
    "DiscretionaryCheckService",
    "check_compliance_v2",
    "check_compliance_v2_bulk",
    "resolve_proportional",
    "AggregateReportService",
    "build_aggregate_report",
    "render_report",
]
