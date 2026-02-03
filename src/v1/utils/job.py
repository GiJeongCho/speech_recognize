from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import threading

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 to 100.0
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class JobManager:
    """
    간단한 In-Memory Job 저장소입니다.
    Production 환경에서는 Redis 등을 사용하여 상태를 영구 저장해야 합니다.
    """
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str) -> JobInfo:
        with self._lock:
            job = JobInfo(job_id=job_id)
            self._jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: float):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.PROCESSING
                job.progress = min(max(progress, 0.0), 100.0)
                if job.started_at is None:
                    job.started_at = datetime.now()

    def complete_job(self, job_id: str, result: Any):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.completed_at = datetime.now()
                job.result = result

    def fail_job(self, job_id: str, error_msg: str):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = error_msg

# Singleton Instance
job_manager = JobManager()

