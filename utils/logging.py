import os
import logging


def open_log(args):
    # open the log file
    log_savepath = os.path.join(args.output_dir)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    # log_name = args.config.split('/')[-1].split('.')[0]+'-yaml'
    log_name = args.name
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))

def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.INFO,
                    format='[%(asctime)s-%(levelname)s] %(message)s',
                    datefmt  = '%y-%m-%d %H:%M:%S',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)