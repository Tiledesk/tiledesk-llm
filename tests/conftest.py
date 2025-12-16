# tests/conftest.py

import pytest
from fastapi.testclient import TestClient
import os
import sys
import fakeredis.aioredis
import asyncio

# --- Add project root to Python path ---
# This ensures that the 'tilellm' module can be imported by the test runner
current_file_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, '..'))
sys.path.insert(0, project_root)
# -----------------------------------------

# Import the app instance
from tilellm.__main__ import app

@pytest.fixture(scope="function")
def client(mocker):
    """
    Pytest fixture to create a FastAPI TestClient with mocked dependencies.

    This fixture uses `pytest-mock` to patch functions that create real 
    external connections during the application's startup lifespan.

    - It patches 'from_url' to return a FakeRedis instance.
    - It patches 'redis_xgroup_create' to do nothing.
    - It patches the 'reader' background task to prevent it from running an
      infinite loop during tests.
    """
    
    # 1. Patch 'from_url' where it's used in __main__ to return a fake client
    mocker.patch(
        'tilellm.__main__.from_url', 
        return_value=fakeredis.aioredis.FakeRedis()
    )

    # 2. Patch the 'redis_xgroup_create' function to prevent it from running
    async def do_nothing(*args, **kwargs):
        pass
        
    mocker.patch(
        'tilellm.__main__.redis_xgroup_create', 
        side_effect=do_nothing
    )
    
    # 3. Patch the 'reader' function itself to prevent the background task
    #    from starting its infinite loop. This is more stable than patching asyncio.
    mocker.patch(
        'tilellm.__main__.reader',
        side_effect=do_nothing
    )

    with TestClient(app) as test_client:
        yield test_client

    # Pytest-mock automatically undoes patches after the fixture goes out of scope

