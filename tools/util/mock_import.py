from typing import Any, Dict, List, Optional, Type, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from traceback import StackSummary, FrameSummary
from tools.util.typing import DEFAULT
def extract_trace(level: Optional[slice] = None) -> StackSummary:
    tb = traceback.extract_stack()
    if level is not None:
        return tb[level]
    return tb
    
@dataclass
class CallItem():

    args: tuple

    kwargs: Dict[str, Any] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=lambda: datetime.now())

    trace: FrameSummary = field(default_factory=lambda: extract_trace(slice(-5, -4))[0])


class MockImport(object):
    """Mock import class. Does nothing but can be used for testing / mocking."""

    __hints__: Dict[Type, Dict[str, Type]] = dict()
    """Stores type hints for mock classes"""

    __declared_modules__ : Dict[str, "MockImport"]

    __call_history__: List[CallItem]

    __silent__: bool 

    __mocked_property__: str

    def __init__(self, silent: bool = False, mocked_property: str = "") -> None:
        self.__declared_modules__ = dict()
        self.__call_history__ = list()
        self.__silent__ = silent
        self.__mocked_property__ = mocked_property

    @classmethod
    def _is_type_hinted_var(cls, name: str) -> bool:
        if cls not in cls.__hints__:
            cls.__hints__[cls] = get_type_hints(cls)
        hints = cls.__hints__[cls]
        return name in hints

    def declare_module(self, module_name: str):
        mi = MockImport(silent=self.__silent__, mocked_property=self.__mocked_property__ + "." + module_name)
        self.__declared_modules__[module_name] = mi
        return mi

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ci = CallItem(args=args, kwargs=kwds)
        self.__call_history__.append(ci)
        return self

    def __getattr__(self, name: str) -> Any:
        if type(self)._is_type_hinted_var(name):
            return object.__getattribute__(self, name)
        else:
            p = self.__declared_modules__.get(name, DEFAULT)
            if p == DEFAULT:
                return self.declare_module(name)
            else:
                return p
        
    def __setattribute__(self, name: str, value: Any) -> None:
        if type(self)._is_type_hinted_var(name):
            object.__setattribute__(self, name, value)
        else:
            self.__declared_modules__[name] = value