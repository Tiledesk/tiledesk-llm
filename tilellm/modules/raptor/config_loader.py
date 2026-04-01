"""
Configuration loader for RAPTOR module.
"""

import os
from typing import Dict, Any, Optional

from tilellm.modules.raptor.models.models import RaptorConfig, RaptorRetrievalStrategy


def load_raptor_config() -> Dict[str, Any]:
    """
    Load RAPTOR configuration from environment variables.
    
    Returns:
        Dictionary with RAPTOR configuration
    """
    config = {
        # Clustering
        "cluster_size": int(os.environ.get("RAPTOR_CLUSTER_SIZE", "5")),
        "max_levels": int(os.environ.get("RAPTOR_MAX_LEVELS", "3")),
        
        # Activation thresholds
        "min_pages_for_raptor": int(os.environ.get("RAPTOR_MIN_PAGES", "10")),
        "enabled_doc_types": os.environ.get(
            "RAPTOR_ENABLED_DOC_TYPES", 
            "accademico,tecnico,legale,scientifico"
        ).split(","),
        
        # Summary generation
        "summary_max_tokens": int(os.environ.get("RAPTOR_SUMMARY_MAX_TOKENS", "512")),
        "summary_temperature": float(os.environ.get("RAPTOR_SUMMARY_TEMPERATURE", "0.3")),
        
        # Retrieval
        "retrieval_strategy": os.environ.get(
            "RAPTOR_RETRIEVAL_STRATEGY", 
            "collapsed_tree"
        ),
        "top_k_per_level": int(os.environ.get("RAPTOR_TOP_K_PER_LEVEL", "3")),
        
        # RRF fusion
        "use_rrf_fusion": os.environ.get("RAPTOR_USE_RRF", "true").lower() == "true",
        "rrf_k": int(os.environ.get("RAPTOR_RRF_K", "60")),
    }
    
    return config


def get_raptor_config_from_env() -> RaptorConfig:
    """
    Create RaptorConfig from environment variables.
    
    Returns:
        RaptorConfig instance
    """
    config_dict = load_raptor_config()
    
    # Convert retrieval_strategy string to enum
    strategy_str = config_dict.pop("retrieval_strategy", "collapsed_tree")
    try:
        strategy = RaptorRetrievalStrategy(strategy_str)
    except ValueError:
        strategy = RaptorRetrievalStrategy.COLLAPSED_TREE
    
    return RaptorConfig(
        **config_dict,
        retrieval_strategy=strategy
    )


def is_raptor_enabled() -> bool:
    """
    Check if RAPTOR module is enabled.
    
    Returns:
        True if RAPTOR is enabled, False otherwise
    """
    enabled = os.environ.get("ENABLE_RAPTOR", "false").lower()
    return enabled in ("true", "1", "yes", "on")


def should_use_raptor_for_document(
    doc_type: Optional[str] = None,
    page_count: Optional[int] = None
) -> bool:
    """
    Determine if RAPTOR should be used for a document based on type and size.
    
    Args:
        doc_type: Document type (accademico, tecnico, etc.)
        page_count: Number of pages in document
        
    Returns:
        True if RAPTOR should be used, False otherwise
    """
    if not is_raptor_enabled():
        return False
    
    config = get_raptor_config_from_env()
    
    # Check page count threshold
    if page_count is not None and page_count < config.min_pages_for_raptor:
        return False
    
    # Check document type
    if doc_type is not None:
        return doc_type.lower() in [t.lower() for t in config.enabled_doc_types]
    
    # If no specific info, default to using RAPTOR if enabled
    return True
