"""
Miscellaneous helper functions and constants.

(c) Nikola Jajcay
"""

import logging
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count

# from pathos.multiprocessing import Pool
from tqdm import tqdm

# project structure
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.abspath(__file__), "..", "..")
)
CODE_ROOT = os.path.join(PROJECT_ROOT, "code")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# # okda, cursa
# NI_SERVER_SIMULATION_DATA = "/mnt/raid/data/jajcay/work-brain/results"
# # furud
# # NI_SERVER_SIMULATION_DATA = "/mnt/scratch/jajcay/work-brain/results"

# get scratch dir from environment
METACENTRUM_SIMULATION_DATA = os.environ.get("SCRATCHDIR")

# various
DATE_FORMAT = "%Y%m%d"
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_EXT = ".log"


def run_in_parallel(
    partial_function,
    iterable,
    workers=cpu_count(),
    length=None,
    assert_ordered=False,
):
    """
    Wrapper for running functions in parallel with tqdm bar.

    :param partial_function: partial function to be evaluated
    :type partial_function: :class:`_functools.partial`
    :param iterable: iterable comprised of arguments to be fed to partial
        function
    :type iterable: iterable
    :param workers: number of workers to be used
    :type workers: int
    :param length: Length of the iterable / generator.
    :type length: int|None
    :param assert_ordered: whether to assert order of results same as the
        iterable (imap vs imap_unordered)
    :type assert_ordered: bool
    :return: list of values returned by partial function
    :rtype: list
    """
    total = length
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            pass

    # wrap method in order to get original exception from a worker process
    partial_function = partial(_worker_fn, fn=partial_function)

    pool = Pool(workers)
    imap_func = pool.imap_unordered if not assert_ordered else pool.imap
    results = []
    for result in tqdm(imap_func(partial_function, iterable), total=total):
        results.append(result)
    pool.close()
    pool.join()

    return results


def _worker_fn(item, fn):
    """
    Wrapper for worker method in order to get original exception from
    a worker process and to log correct exception stacktrace.

    :param item: item from iterable
    :param fn: partial function to be evaluated
    :type fn: :class:`_functools.partial`
    """
    try:
        return fn(item)
    except Exception as e:
        logging.exception(e)
        raise


def make_dirs(path):
    """
    Create directory.

    :param path: path for new directory
    :type path: str
    """
    try:
        os.makedirs(path)
    except OSError as error:
        logging.warning(f"{path} could not be created: {error}")


def today():
    """
    Return formatted today's date.

    :return: today's date
    :rtype: str
    """
    return datetime.today().strftime(DATE_FORMAT)


def set_logger(log_filename=None, log_level=logging.INFO):
    """
    Prepare logger.

    :param log_filename: filename for the log, if None, will not use logger
    :type log_filename: str|None
    :param log_level: logging level
    :type log_level: int
    """
    formatting = "[%(asctime)s] %(levelname)s: %(message)s"
    log_formatter = logging.Formatter(formatting, LOG_DATETIME_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    # set terminal logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # set file logging
    if log_filename is not None:
        if not log_filename.endswith(LOG_EXT):
            log_filename += LOG_EXT
        logging.warning(f"{log_filename} already exists, removing...")
        if os.path.exists(log_filename):
            os.remove(log_filename)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)


if __name__ == "__main__":
    set_logger(log_filename=None)
    logging.info(f"Project root: {PROJECT_ROOT}")
    logging.info(f"Code folder: {CODE_ROOT}")
    logging.info(f"Data folder: {DATA_ROOT}")
    logging.info(f"Experiments folder: {EXPERIMENTS_ROOT}")
    if not os.path.exists(EXPERIMENTS_ROOT):
        make_dirs(EXPERIMENTS_ROOT)
    logging.info(f"Results folder: {RESULTS_ROOT}")
    if not os.path.exists(RESULTS_ROOT):
        make_dirs(RESULTS_ROOT)
