import pytest
from tools.util.format import raise_on_none
from tools.error.argument_none_error import ArgumentNoneError
from tools.error.argument_none_type_suggestion_error import ArgumentNoneTypeSuggestionError
from typing import Any


def test_raise_on_none_raising():
    def some_func(my_var: Any):
        my_var = raise_on_none(my_var)
        return my_var
    with pytest.raises(ArgumentNoneError) as e:
        some_func(None)
        pass
    assert e.type == ArgumentNoneError


def test_raise_on_none_not_raising():
    def some_func(my_var: Any):
        my_var = raise_on_none(my_var)
        return my_var
    assert some_func(1) == 1
    assert some_func("Hello") == "Hello"
    assert some_func({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def my_func(my_var: Any):
    my_var = raise_on_none(my_var)
    return my_var


def test_raise_on_none_raising_with_type():
    with pytest.raises(ArgumentNoneError) as e:
        my_func(None)
        pass
    assert e.type == ArgumentNoneTypeSuggestionError
