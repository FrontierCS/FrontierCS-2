import logging
import sys


class ColorizedFormatter(logging.Formatter):
    color_map = {
        'black': "\033[0;30m",
        'red': "\033[0;31m",
        'bold_red': "\033[1;31m",
        'green': "\033[0;32m",
        'yellow': "\033[1;33m",
        'blue': "\033[0;34m",
        'reset': "\033[0m",
    }

    level_to_color = {
        logging.DEBUG: 'reset',
        logging.INFO: 'green',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'bold_red'
    }

    def color_formatter(self, color):
        fmt = "\r[%(asctime)s] [%(levelname)s] %(message)s"
        datefmt = "%H:%M:%S"

        formatter = logging.Formatter(self.color_map[color] + fmt + self.color_map['reset'], datefmt)
        return formatter

    def format(self, record):
        return self.color_formatter(self.level_to_color[record.levelno]).format(record)


class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


logger = logging.getLogger('logger')
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter("\r[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

handler.addFilter(InfoOnlyFilter())

handler.setFormatter(ColorizedFormatter())
logger.addHandler(handler)


if __name__ == '__main__':
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')
