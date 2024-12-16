"""
  @FileName：log_config.py
  @Author：Excelius
  @CreateTime：2024/11/9 21:45
  @Company: None
  @Description：
"""
import logging
import os

# 创建日志文件夹（如果不存在）
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# 设置日志配置
def setup_logger(name: str):
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 清除现有的所有处理器，确保新的配置生效
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 如果需要可以提供一个全局的logger
logger = setup_logger('app_logger')
