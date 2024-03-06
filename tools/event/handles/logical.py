
import logging
from typing import Any, Callable, Dict, Optional, Type, Union
import os


def or_(*handles: Callable[[Dict[str, Any], Any], bool]) -> Callable[[Dict[str, Any], Any], bool]:
    """
    Gets bool evaluating event handles and perform a logical or on their return.

    May not execute all handles.

    Parameters
    ----------
    handles : Callable[[Dict[str, Any], Any], bool]
        The handles to execute.

    Returns
    -------
    Callable[[Dict[str, Any], Any], bool]
        The handle which executes all handles passed as arguments and perform a logical or on their return.
    """
    def _handle_or(ctx: Dict[str, Any], args: Any):
        for handle in handles:
            res = handle(ctx, args)
            if res is True:
                return True
        return False
    return _handle_or


def and_(*handles: Callable[[Dict[str, Any], Any], bool]) -> Callable[[Dict[str, Any], Any], bool]:
    """
    Gets bool evaluating event handles and perform a logical and on their return.
    
    If at least one handle evaluates to a False, the handle will return False.
    If all handles evaluate to a True, the handle will return True.

    May not execute all handles.

    Parameters
    ----------
    handles : Callable[[Dict[str, Any], Any], bool]
        The handles to execute.

    Returns
    -------
    Callable[[Dict[str, Any], Any], bool]
        The handle which executes all handles passed as arguments and perform a logical and on their return.
    """
    def _handle_and(ctx: Dict[str, Any], args: Any):
        for handle in handles:
            res = handle(ctx, args)
            if res is False:
                return False
        return True
    return _handle_and

def not_(handle: Callable[[Dict[str, Any], Any], bool]) -> Callable[[Dict[str, Any], Any], bool]:
    """
    Gets a bool evaluating event handle and perform a logical not on its return.

    Parameters
    ----------
    handles : Callable[[Dict[str, Any], Any], bool]
        The handle to execute.

    Returns
    -------
    Callable[[Dict[str, Any], Any], bool]
        The handle which executes the original handle and performs a not on its return.
    """
    def _handle_and(ctx: Dict[str, Any], args: Any):
        return not handle(ctx, args)   
    return _handle_and