import logging
from pathlib import Path
from datetime import datetime


class LoggerSetup:
    def __init__(
            self, 
            module_name: str, 
            level: int = logging.DEBUG, 
            log_file_path: str | None = None
        ):
        """
        Configure a logger with console output and optional file output.
        """
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(level)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        fmt = "[%(asctime)s] - [%(name)s] - [%(funcName)s():%(lineno)d %(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        self.console_format = logging.Formatter(fmt, datefmt)
        self.console_handler.setFormatter(self.console_format)
        self.logger.addHandler(self.console_handler)

        self.logger.propagate = False

        # Determine log file path: default to ./logs/<YYYY-mm-dd>.log
        log_path = self._resolve_log_path(log_file_path)
        if log_path is not None:
            # Ensure the directory exists
            log_dir = log_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)

            self.file_handler = logging.FileHandler(log_path)
            self.file_handler.setLevel(level)
            self.file_handler.setFormatter(self.console_format)
            self.logger.addHandler(self.file_handler)

    def _resolve_log_path(self, log_file_path: str | None) -> Path | None:
        """
        Decide the log file path based on provided input.

        Rules:
        - If log_file_path is None: use ./logs/<YYYY-MM-DD>.log
        - If log_file_path points to a directory (endswith '/' or existing dir): place <YYYY-MM-DD>.log inside it
        - Otherwise, treat log_file_path as a file path
        """
        default_filename = f"{datetime.now().strftime('%Y-%m-%d')}.log"

        if log_file_path is None:
            return Path("./logs") / default_filename

        candidate = Path(log_file_path)
        # If explicit directory or path string ends with a slash
        if str(log_file_path).endswith("/") or candidate.exists() and candidate.is_dir():
            return candidate / default_filename

        # If parent is missing, we'll create it before using as file path
        return candidate


if __name__ == "__main__":
    logger = LoggerSetup(module_name="main").logger
    logger.info("Logger initialized in main execution context")
    logger.debug("Logger initialized in main execution context")
    logger.warning("Logger initialized in main execution context")


