import numpy as np
import pytest

from ccounts.checks import validIdentifier


@pytest.mark.parametrize(
    "value,expected",
    [
        ("cell-1:ACGT", True),
        ("\u03b1", True),
        (np.str_("cell-1"), True),
        ("", False),
        (None, False),
        (1, False),
        (b"cell-1", False),
        ("cell\0-1", False),
        ("cell 1", False),
        ("cell\t1", False),
        ("cell\n1", False),
        ("cell\u20031", False),
    ],
)
def testIdentifierValidation(value, expected):
    assert validIdentifier(value) is expected
