import logging
from typing import List, Dict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Query(BaseModel):
    id: str
    requests: str = ""

class FreeTextQuery(BaseModel):
    id: str
    text: str = ""


class ResultItem(BaseModel):
    sku: str
    description: str = ""
    quantity: Dict = {}
    errors: List[str] = []


class ResultOrder(BaseModel):
    originalRequest: str = ""
    order: List[ResultItem] = []
    additionalFields: Dict = {}
    errors: List[str] = []


class Result(BaseModel):
    id: str
    processingResult: List[ResultOrder] = []
    errors: List[str] = []
