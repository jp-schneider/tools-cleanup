import threading


def debounce(wait_time: float, default_enabled: bool = True):
    """
    Decorator that will debounce a function so that it is called after wait_time seconds
    If it is called multiple times, will wait for the last call to be debounced and run only this one.
    
    If the function is called with debounce=False, it will be called immediately.

    Parameters
    ----------
    wait_time : float
        Time to wait before calling the function.

    default_enabled : bool
        Whether the debounce is enabled by default. If False, the function will be called immediately, unless called with debounce=True.
        Default is True.

    Returns
    -------
    decorator
        Decorator that will debounce a function so that it is called after wait_time seconds
    """
    def decorator(function):
        def debounced(*args, **kwargs):
            db = kwargs.pop("debounce", default_enabled)
            # if the function is called with debounce=False, call it immediately
            if not db:
                if debounced._timer is not None:
                    debounced._timer.cancel()
                    debounced._timer = None
                return function(*args, **kwargs)
            
            def call_function():
                debounced._timer = None
                return function(*args, **kwargs)
            # if we already have a call to the function currently waiting to be executed, reset the timer
            if debounced._timer is not None:
                debounced._timer.cancel()

            # after wait_time, call the function provided to the decorator with its arguments
            debounced._timer = threading.Timer(wait_time, call_function)
            debounced._timer.start()

        debounced._timer = None
        return debounced
    return decorator