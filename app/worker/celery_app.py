import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "llm_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    # Avoid reserving multiple expensive jobs while a model is busy.
    worker_prefetch_multiplier=1,
    result_expires=3600,
    # HTTP submission/status handlers must fail promptly if Redis is unavailable.
    broker_connection_timeout=3,
    broker_transport_options={
        "socket_connect_timeout": 3,
        "socket_timeout": 3,
        "retry_on_timeout": False,
        "max_retries": 0,
    },
    task_publish_retry=False,
    broker_connection_retry_on_startup=True,
    redis_socket_connect_timeout=3,
    redis_socket_timeout=3,
    redis_retry_on_timeout=False,
    result_backend_always_retry=False,
    result_backend_max_retries=0,
    result_backend_transport_options={
        "retry_policy": {"max_retries": 0},
    },
)
