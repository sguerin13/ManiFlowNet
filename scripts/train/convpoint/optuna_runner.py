import os
import subprocess
import warnings

import requests

from scripts.helpers import load_config
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    config = load_config(os.path.join("scripts","train","convpoint","config","optuna_runner.json"))
    runs_to_generate = config.searchSteps
    
    successful_runs = 0
    fail_count = 0
    main = os.path.join("scripts","train","convpoint","optuna_routine.py")
    while successful_runs < runs_to_generate:
        
        try:
            result = subprocess.run(["python",main])
            print(result)
            successful_runs += 1

        except:
            print("iteration ",successful_runs, " failed")
            fail_count += 1

    if config.includeText:
        r = requests.post('https://textbelt.com/text',json={
            'number': config.phoneNumber,
            'message': "hparam tuning complete",
            'key':config.textBeltKey
        })