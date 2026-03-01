"""Set, SonicIdentity, and Prop CRUD API routes.

Sets are location/environment entities within a Production Bible with visual
references, reverse prompts, and sonic identity sub-entities.
Props are physical objects used in production.

Sub-entities:
  - SonicIdentity (1:1 per Set) — ambient audio characteristics
  - Reference image upload with LLM Vision reverse-prompting for Sets

Spec reference: Phase 17 - PBEX-07, PBEX-08, PBEX-09, PBEX-13, PBEX-14
"""

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, select

from vidpipe.db import async_session
from vidpipe.db.models import ProductionBible, Prop, Set, SonicIdentity
from vidpipe.services.storage_backend import get_storage_backend, LocalStorageBackend

logger = logging.getLogger(__name__)

sets_props_router = APIRouter(prefix="/api")

# Placeholder: full implementation in Task 2
