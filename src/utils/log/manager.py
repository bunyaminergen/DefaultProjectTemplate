import os
import logging
from typing import Annotated, Optional
from logging.handlers import RotatingFileHandler

from pydantic import ValidationError

from src.utils.type.schema import LoggerType, SingleLineConsoleFormatterType, SingleLineFileFormatterType


class SingleLineConsoleFormatter(logging.Formatter):
    def __init__(
            self,
            app: Annotated[str, "Application name"],
            date_format: Annotated[Optional[str], "Date format"] = None,
    ) -> None:
        tmp_logger = logging.getLogger(__name__)

        try:
            config = SingleLineConsoleFormatterType(app=app, date_format=date_format)
        except ValidationError as e:
            tmp_logger.error("Invalid parameters for SingleLineConsoleFormatter: %s", e.json())
            raise

        self.app = config.app
        self.date_format = config.date_format

        console_format = "%(asctime)s - %(app)s - %(levelname)s - %(name)s - %(message)s"
        super().__init__(fmt=console_format, datefmt=self.date_format)

    def format(self, record: logging.LogRecord) -> str:
        record.app = self.app
        return super().format(record)


class SingleLineFileFormatter(logging.Formatter):
    def __init__(
            self,
            app: Annotated[str, "Application name"],
            date_format: Annotated[Optional[str], "Date format"] = None,
    ) -> None:
        tmp_logger = logging.getLogger(__name__)

        try:
            config = SingleLineFileFormatterType(app=app, date_format=date_format)
        except ValidationError as e:
            tmp_logger.error("Invalid parameters for SingleLineFileFormatter: %s", e.json())
            raise

        self.app = config.app
        self.date_format = config.date_format

        file_format = "%(asctime)s - %(app)s - %(levelname)s - %(name)s - %(message)s"
        super().__init__(fmt=file_format, datefmt=self.date_format)

    def format(self, record: logging.LogRecord) -> str:
        record.app = self.app
        return super().format(record)


class Logger:
    def __init__(
            self,
            name: Annotated[str, "Logger name"],
            app: Annotated[str, "Application name"],
            path: Annotated[str, "Folder to store log files"] = ".logs",
            file: Annotated[str, "Log file name"] = ".log",
            console_level: Annotated[int, "Console log level"] = logging.DEBUG,
            file_level: Annotated[int, "File log level"] = logging.DEBUG,
            max_bytes: Annotated[int, "Max file size for rotation"] = 5_000_000,
            backup_count: Annotated[int, "Number of old logs to keep"] = 5,
            verbose: Annotated[bool, "Whether logs are printed to console"] = True,
    ) -> None:

        tmp_logger = logging.getLogger(__name__)

        try:
            settings = LoggerType(
                name=name,
                app=app,
                path=path,
                file=file,
                console_level=console_level,
                file_level=file_level,
                max_bytes=max_bytes,
                backup_count=backup_count,
                verbose=verbose,
            )
        except ValidationError as e:
            tmp_logger.error("Invalid Logger parameters: %s", e.json())
            raise

        self.name = settings.name
        self.app = settings.app
        self.path = settings.path
        self.file = settings.file
        self.console_level = settings.console_level
        self.file_level = settings.file_level
        self.max_bytes = settings.max_bytes
        self.backup_count = settings.backup_count
        self.verbose = settings.verbose

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            self._setup()

    def _setup(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        log_file_path = os.path.join(self.path, self.file)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        file_handler.setLevel(self.file_level)
        file_formatter = SingleLineFileFormatter(
            date_format="%Y-%m-%d %H:%M:%S",
            app=self.app,
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_formatter = SingleLineConsoleFormatter(
                date_format="%Y-%m-%d %H:%M:%S",
                app=self.app,
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def get(self) -> logging.Logger:
        return self.logger


if __name__ == "__main__":
    log_manager = Logger(app="TestApp", name="TestLogger")
    logger = log_manager.get()

    logger.debug("Application debug log.")
    logger.info("Application is starting.")
    logger.warning("Warning! Something unexpected happened.")
    logger.error("An error has occurred!")
    logger.critical("This is a critical message.")
