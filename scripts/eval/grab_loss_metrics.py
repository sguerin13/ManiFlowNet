import os
from tbparse import SummaryReader
from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(os.path.join("scripts","eval","grab_loss_metrics.json"))
    log_dir = config.logDir
    reader = SummaryReader(log_dir)
    df = reader.scalars
    print(df)
