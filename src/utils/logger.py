import structlog

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)

log = structlog.get_logger()