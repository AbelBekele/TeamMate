# job_wait_time.py
import random
import time

def get_current_wait_time(job: str) -> int | str:
    """Dummy function to generate fake wait times"""

    if job not in ["Software Engineer", "B", "C", "D"]:
        return f"job wait {job} does not exist"

    # Simulate API call delay
    time.sleep(1)

    return random.randint(0, 10000)
