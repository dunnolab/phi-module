import json
from datetime import datetime


class BasicLogger:
    def __init__(self, name="DefaultLogger", log_file="../logs/log.json", verbose=True):
        self.name = name
        self.log_file = log_file
        self.verbose = verbose

        with open(self.log_file, "w") as f:
            json.dump([], f, indent=4)
        if self.verbose:
            print(f"Logger '{self.name}' initialized. Logs will be saved to '{self.log_file}'.")

    def log(self, metrics: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }

        with open(self.log_file, "r+") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=4)

        if self.verbose:
            timestamp = datetime.now().isoformat()
            print(f"[{self.name}] Log Entry @ {timestamp}:\n" + \
                    "\n".join(f"          {key}: {value}" for key, value in metrics.items()) + "\n")



