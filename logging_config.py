import logging
import os
from datetime import datetime

# ANSI color codes
class Colors:
    RED = '\033[91m'      # Bright red for errors
    YELLOW = '\033[93m'   # Bright yellow for warnings
    BLUE = '\033[94m'     # Bright blue for debug system messages
    GREEN = '\033[92m'    # Bright green for info
    GREY = '\033[90m'     # Grey for LLM outputs and less important info
    RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to system messages in logs"""
    
    def format(self, record):
        if not isinstance(record.msg, str):
            return super().format(record)
            
        # First, determine if this is an LLM response message
        is_llm_output = False
        if any(marker in record.msg.lower() for marker in [
            "response was:", 
            "result:", 
            "extracted sentence:", 
            "\n{", 
            "response:\n"
        ]):
            is_llm_output = True
            
        # Add color to the system message part (before any variable content)
        if ":" in record.msg:
            system_msg, var_content = record.msg.split(":", 1)
            
            # Choose color based on message level and content
            if is_llm_output:
                # For LLM outputs, color the system part normally but grey out the content
                if record.levelno == logging.ERROR:
                    color = Colors.RED
                elif record.levelno == logging.WARNING:
                    color = Colors.YELLOW
                elif record.levelno == logging.INFO:
                    color = Colors.GREEN
                else:
                    color = Colors.BLUE
                record.msg = f"{color}{system_msg}{Colors.RESET}:{Colors.GREY}{var_content}{Colors.RESET}"
            else:
                # For system messages, color normally
                if record.levelno == logging.ERROR:
                    color = Colors.RED
                elif record.levelno == logging.WARNING:
                    color = Colors.YELLOW
                elif record.levelno == logging.INFO:
                    color = Colors.GREEN
                else:
                    color = Colors.BLUE
                record.msg = f"{color}{system_msg}{Colors.RESET}:{var_content}"
        
        return super().format(record)

def setup_logging(log_dir="logs"):
    """
    Sets up logging configuration for the project.
    Creates separate log files for preprocessing and correction steps.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocess_log = os.path.join(log_dir, f"preprocessing_{timestamp}.log")
    correction_log = os.path.join(log_dir, f"correction_{timestamp}.log")
    
    # Set up preprocessing logger
    preprocess_logger = logging.getLogger('preprocessing')
    preprocess_logger.setLevel(logging.DEBUG)
    
    # File handler with standard formatter (no colors in file)
    fh = logging.FileHandler(preprocess_log)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    preprocess_logger.addHandler(fh)
    
    # Set up correction logger
    correction_logger = logging.getLogger('correction')
    correction_logger.setLevel(logging.DEBUG)
    
    # File handler for correction logger
    fh = logging.FileHandler(correction_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    correction_logger.addHandler(fh)
    
    # Console handler with colored formatter
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    colored_formatter = ColoredFormatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(colored_formatter)
    
    preprocess_logger.addHandler(console)
    correction_logger.addHandler(console)
    
    return preprocess_logger, correction_logger 