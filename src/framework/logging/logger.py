import sys
from os.path import join


class Logger:

    def __init__(self):
        self.terminal = sys.stdout
        self.log_path = 'data/logging/'

    def write(self, message):
        with open(join(self.log_path, "logfile.log"), "w", encoding='utf-8') as self.log:
            self.log.write(message)

        self.terminal.write(message)

    def flush(self):
        pass

