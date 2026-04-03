
import logging
import os
from datetime import datetime

def get_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 添加时间戳到日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(save_path, f'log_{timestamp}.txt')
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger