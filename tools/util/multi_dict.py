class MultiKey(frozenset):
    """A MultiKey indicates that multiple keys are part of the parameter grid."""
    pass


class MultiValue(dict):
    """A MultiValue indicates that multiple values are part of the parameter grid."""
    pass


class MultiDict(dict):
    pass