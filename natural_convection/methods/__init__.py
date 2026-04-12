"""Реестр расчётных методик."""

from .leontiev import calc_leontiev
from .churchill_ozoe import calc_churchill_ozoe
from .vliet import calc_vliet
from .fujii import calc_fujii
from .isachenko import calc_isachenko

ALL_METHODS = {
    'leontiev': calc_leontiev,
    'churchill_ozoe': calc_churchill_ozoe,
    'vliet': calc_vliet,
    'fujii': calc_fujii,
    'isachenko': calc_isachenko,
}
