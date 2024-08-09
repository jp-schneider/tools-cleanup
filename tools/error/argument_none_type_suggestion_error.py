from tools.error.argument_none_error import ArgumentNoneError


class ArgumentNoneTypeSuggestionError(ArgumentNoneError):
    """Argument None Type Suggestion Error for creating an exception with just the argument name and the suggested type.
    Will print the suggested type in the error message.
    """

    def __init__(self, argument_name: str, suggested_type: type):
        super().__init__(argument_name,
                         f"The value should be of type: {suggested_type}")
