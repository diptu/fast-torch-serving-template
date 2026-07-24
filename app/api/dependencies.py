from typing import Annotated

from fastapi import Depends

from app.services.inference_service import InferenceService, get_inference_service

InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
