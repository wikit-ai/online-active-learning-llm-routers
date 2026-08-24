import logging


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/healthz") == -1


def get_logger(name: str):
    app_logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '{"level": "%(levelname)s", "message": "%(message)s", "time":' ' "%(asctime)s"}'
    )
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.DEBUG)
    app_logger_adapted = logging.LoggerAdapter(app_logger, {"request_id": "-"})
    return app_logger_adapted


logger = get_logger(__name__)
