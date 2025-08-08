import time
from datetime import datetime, timedelta
from tools.util.format import strfdelta
from typing import List, Any, Optional


class Timer:
    """Time class which can be used as context manager to measure times.

    >>> with Timer() as timer:
            ...
        print(timer.duration)

    Will result int the time as timedelta.
    """

    _start: float
    """Start time of timer."""

    _end: float
    """End time of timer"""

    _pauses: List["TimerPause"]
    """Number of pauses of the timer."""

    def __init__(self) -> None:
        self._start = -1
        self._end = -1
        self._pauses = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def pause(self) -> "TimerPause":
        """Returns a timer pause object which can be used to pause the timer, when used as context manager.

        E.g.:
        >>> with Timer() as timer:
                with timer.pause():
                    # do something
            print(timer.duration)
        """
        # Check if pausing is possible
        if len(self._pauses) > 0 and self._pauses[-1]._start != -1 and self._pauses[-1]._end == -1:
            raise ValueError(
                "Timer is already paused, you can not pause it again.")
        pause = TimerPause()
        self._pauses.append(pause)
        return pause

    def start(self) -> float:
        """Starts the timer.
        Can be used in manual mode.

        Returns
        -------
        float
            The time in UTC Timestamp when it was started.
        """
        self._start = time.time()
        return self._start

    def stop(self) -> float:
        """Stops the timer.
        Can be used in manual mode.

        Returns
        -------
        float
            The time in UTC Timestamp when it was stopped.
        """
        self._end = time.time()
        return self._end

    @property
    def start_date(self) -> datetime:
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        return datetime.fromtimestamp(self._start)

    @property
    def end_date(self) -> datetime:
        if self._end == -1:
            raise ValueError(f"Timer was not stopped!")
        return datetime.fromtimestamp(self._end)

    @property
    def duration(self) -> timedelta:
        """The duration between start and stop.
        Raises an error when it was not started / stopped.

        Pause duration are subtracted from the total duration.

        Returns
        -------
        timedelta
            The duration of the measured time.

        Raises
        ------
        ValueError
            If the timer was not started or stopped.
        """
        return self._compute_duration(total=False)

    @property
    def total_duration(self) -> timedelta:
        """The total duration between start and stop.
        Raises an error when it was not started / stopped.

        Pause duration are not subtracted from the total duration.

        Returns
        -------
        timedelta
            The total duration of the measured time.

        Raises
        ------
        ValueError
            If the timer was not started or stopped.
        """
        return self._compute_duration(total=True)

    @classmethod
    def strfdelta(cls, timedelta: timedelta, format: str = "%D days %H:%M:%S.%f") -> str:
        return strfdelta(timedelta, format)

    def _compute_duration(self, total: bool = False) -> timedelta:
        """Computes the duration of the timer.
        Raises an error when it was not started / stopped.

        Returns
        -------
        timedelta
            The duration of the measured time.

        Raises
        ------
        ValueError
            If the timer was not started or stopped.
        """
        self._check_values()
        total_duration = self._end - self._start
        if total:
            return timedelta(seconds=total_duration)
        else:
            # Subtract all pauses
            ps = []
            for pause in self._pauses:
                try:
                    ps.append(pause.elapsed(
                        total=True, end=self._end).total_seconds())
                except ValueError:
                    pass
            return timedelta(seconds=total_duration - sum(ps))

    def elapsed(self, total: bool = False, end: Optional[float] = None) -> timedelta:
        """Returns the elapsed time.
        Can also be accessed when timer is not stopped.

        Returns
        -------
        timedelta
            The elapsed time since start.

        Raises
        ------
        ValueError
            If the timer was not started.
        """
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        _end = self._end if self._end != - \
            1 else (time.time() if end is None else end)
        if total:
            return timedelta(seconds=_end - self._start)
        else:
            # Subtract all pauses
            ps = []
            for pause in self._pauses:
                try:
                    ps.append(pause.elapsed(
                        total=True, end=_end).total_seconds())
                except ValueError:
                    pass
            return timedelta(seconds=_end - self._start - sum(ps))

    def _check_values(self):
        if self._start == -1:
            raise ValueError(f"Timer was not started!")
        if self._end == -1:
            raise ValueError(f"Timer was not stopped!")


class TimerPause(Timer):
    """Timer pause class which can be used as context manager to measure times with pauses."""

    def pause(self) -> "TimerPause":
        raise ValueError(
            "TimerPause can not be paused, it is already a pause.")
