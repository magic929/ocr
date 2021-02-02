import logging

class Logger():
    def __init__(self, logfile, logins, loglv):
        logging.basicConfig(filename=logfile, filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
        self.log = logging.getLogger(logins)
    
    def debug(self, msg):
        self.log.debug(msg)
    
    def info(self, msg):
        self.log.info(msg)

    def warn(self, msg):
        self.log.warn(msg)
    
    def error(self, msg):
        self.log.error(msg)
    
    def critical(self, msg):
        self.log.critical(msg)