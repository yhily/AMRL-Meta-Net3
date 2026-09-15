import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", filename=None):
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_{timestamp}.csv"

        self.path = os.path.join(log_dir, filename)
        self.file = open(self.path, "w", newline="")
        self.writer = None

    def write(self, data: dict):
        """
        data: dictionary of scalar values
        """
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=data.keys())
            self.writer.writeheader()

        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()
