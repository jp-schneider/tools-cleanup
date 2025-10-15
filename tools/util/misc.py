import os


def get_max_pid() -> int:
    """Get the maximum PID value for the current system."""
    if os.name == "nt":
        return 2 ** 32 - 1  # Windows typically has a theoretical max PID of 2 ** 32 - 1
    else:
        try:
            with open("/proc/sys/kernel/pid_max", "r") as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return 4194304  # A common default max PID on many Linux systems
