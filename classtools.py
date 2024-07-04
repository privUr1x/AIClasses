#!/usr/bin/python3
from typing import Any, Iterable, Sized, Tuple, Type, Union, Callable, TypeVar
from inspect import signature, getmembers

Compound = TypeVar("Compound")

class Tool:
    def __init__(self) -> None:
        self._info = ""

    @property
    def info(self) -> str:
        """Gets the info value."""
        return self._info

    @info.setter
    def info(self, value: str) -> None:
        """Function to attach a value to self.info."""
        if not isinstance(value, str):
            raise TypeError("Expected info to be str.")

        self._info = value

class Verifiers(Tool):
    """Class representing uniform verifiers for objects."""

    def __init__(self) -> None:
        super().__init__()
        self.info = "Meant to be used to simplify commonly used conditionals."

    @staticmethod
    def verify_type(obj: Any, t: Union[type, Tuple[type, ...]]) -> Any:
        """
        Verifies if obj is an instance of a given type
        Raises:
            - TypeError: raises if the object is not the expected type.
        """
        if not isinstance(obj, t):
            raise TypeError(f"Expected {obj} to be {t}, got type {type(obj)}.")

        return obj

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class Testers(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.info = "Meant to be used to simplify testing a new class."
        self.__dependencies = ["re", "inspect"]
        self.__import_dependencies()

    def __import_dependencies(self) -> None:
        for module_name in self.__dependencies:
            try:
                # Import the module
                __import__(module_name)
            except ImportError as e:
                print(f"[!] Error: {e}")

    @staticmethod
    def test_methods(clss: Type):
        raise NotImplemented
        # Acceding class methods

        # Verifing if methods are Callable

        # Removing special attributes

        # Creating a personalized test for every method depending on arguments types
        pass

    @staticmethod
    def test_method(func: Callable, n: int = 5) -> bool:
        raise NotImplemented
        Verifiers.verify_type(n, int)
        if not callable(func):
            raise TypeError("Expected callable argument for func.")

        # Obtaining func signature
        sign = signature(func)

        # Looping through parameters and annotations
        args = {
            param_name: param.annotation
            for param_name, param in sign.parameters.items()
        }

        print(args)

        # Print errors

        return False
