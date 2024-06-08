import os
from functools import wraps

import logging

logger = logging.getLogger(__name__)


def inject_repo(func):
    """
    Annotation for inject PineconeRepository.
    If PINECONE_TYP is pod is injected PineconeRepositoryPod
    If PINECONE_TYP is serverless is injected PineconeRepositoryServerless
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        repo_type = os.environ.get("PINECONE_TYPE")
        logger.info(f"pinecone type {repo_type}")
        if repo_type == 'pod':

            from tilellm.store.pinecone_repository_pod import PineconeRepositoryPod
            repo = PineconeRepositoryPod()
        elif repo_type == 'serverless':
            from tilellm.store.pinecone_repository_serverless import PineconeRepositoryServerless
            repo = PineconeRepositoryServerless()
        else:
            raise ValueError("Unknown repository type")

        kwargs['repo'] = repo
        return func(*args, **kwargs)

    return wrapper
