import logging

class CustomLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        logging.basicConfig(level=logging.INFO)

    def info(self, msg, **kwargs):
        # Format the extra arguments into the message string
        if kwargs:
            msg = f"{msg} | Details: {kwargs}"
        self.logger.info(msg)

def get_logger(name):
    return CustomLogger(name)