import sys
from networkSecurity.logging import logger
class NetworkSecurityException(Exception):
    def __init__(self,message,details:sys):
        self.message = message
        _,_,exc_tb = details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(self.file_name, self.lineno, self.message)
    
# if __name__ == "__main__":
#     logger.logging.info()