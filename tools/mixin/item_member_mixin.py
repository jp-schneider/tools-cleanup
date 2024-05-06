class ItemMemberMixin:
    """Simple mixin to allow for getting class members using the __getitem__ and __setitem__ method."""

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise KeyError(f"Property: {item} must be a string")
        return getattr(self, item)

    def __setitem__(self, key: str, value):
        if not isinstance(key, str):
            raise KeyError(f"Property: {key} must be a string")
        setattr(self, key, value)
