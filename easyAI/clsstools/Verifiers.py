from typing import Any, Iterable, Sized, Tuple, Union, Protocol

class Compound(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...

def verify_type(obj: Any, t: Union[type, Tuple]) -> Any:
    """
    Verifies if obj is an instance of a given type
    Raises:
        - TypeError: raises if the object is not the expected type.
    """
    if not isinstance(obj, t):
        raise TypeError(f"Expected {obj} to be {t}, got type {type(obj)}.")

    return obj

def verify_len(obj: Sized, n: int) -> Any:
    """
    Verifies the length of an object (o).
    Raises:
        - IndexError: raises if the object is different in length as expected.
    """
    if hasattr(obj, "__len__"):
        if len(obj) != n:
            raise IndexError(f"Expected {obj} to be {n} in length, got length {len(obj)}.")

    return obj

def verify_indexable(obj: object) -> Any:
    """
    Check if an object is indexable (has __getitem__ method).

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is indexable, False otherwise.
    """
    if not hasattr(obj, "__getitem__") and callable(getattr(obj, "__getitem__")):
        raise IndexError("Not supported.")

    return obj

def verify_iterable(obj: Iterable) -> Any:
    """
    Verifies if an object is iterable.
    Raises:
        - TypeError: raises if the object is not iterable.
    """
    try:
        iter(obj)
    except TypeError:
        raise TypeError(f"The object of type {type(obj).__name__} is not iterable")

    return obj

def verify_components_type(obj: Compound, etype: Union[type, Tuple[type, ...]]) -> Any:
    """
    Check if an object has the correct containing type.
    Object must have __len__ and __getitem__ methods defined.

    Args:
        obj (Any): The object to check.
        etype type: The expected components type(s).

    Returns:
        type(obj): the given object
    """
    for i in range(len(obj)):
        if not isinstance(obj[i], etype):
            raise TypeError(f"Expected {obj} to have {etype} components, got {type(obj[i])} for {obj[i]} on position {i}.")

    return obj
