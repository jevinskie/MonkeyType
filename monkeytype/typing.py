# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from __future__ import annotations

import functools
import importlib
import inspect
import types
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    DefaultDict,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    ParamSpec,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import TypedDict

from monkeytype.compat import (
    is_any,
    is_generic,
    is_generic_of,
    is_typed_dict,
    is_union,
    name_of_generic,
    types_equal,
)

if not TYPE_CHECKING:
    try:
        from rich import print
    except ImportError:
        pass


DUMMY_TYPED_DICT_NAME = "DUMMY_NAME"
DUMMY_REQUIRED_TYPED_DICT_NAME = "REQUIRED_TYPED_DICT_NAME"
DUMMY_OPTIONAL_TYPED_DICT_NAME = "OPTIONAL_TYPED_DICT_NAME"


_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])
_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class NamePath(NamedTuple):
    module: str
    qualname: str


class ResolvedNamePath(NamedTuple):
    namepath: NamePath
    module: types.ModuleType
    value: Any


class AnnotatedMethodInfo(NamedTuple):
    resolved: ResolvedNamePath
    name: str
    method: types.MethodType


AMI = AnnotatedMethodInfo
AMIS = cast(AnnotatedMethodInfo, object())


def dotted_getattr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_namepath(np: NamePath) -> ResolvedNamePath:
    mod = importlib.import_module(np.module)
    val = dotted_getattr(mod, np.qualname)
    return ResolvedNamePath(np, mod, val)


def get_namepath(val: Any) -> NamePath:
    if not hasattr(val, "__module__"):
        raise ValueError(f"Can't get NamePath: __module__ missing from val: {val}")
    if not hasattr(val, "__qualname__"):
        raise ValueError(f"Can't get NamePath: __qualname__ missing from val: {val}")
    return NamePath(val.__module__, val.__qualname__)


class AnnotatedMethod(Generic[_T, _P, _R_co]):
    _rnp: ResolvedNamePath
    _n: str
    _f: Callable[Concatenate[_T, _P], _R_co]
    _fmeta: Callable[Concatenate[_T, _P], _R_co]

    # FIXME: Need weakref?

    def __init__(self, func: Callable[Concatenate[_T, _P], _R_co], namepath: NamePath) -> None:
        self._rnp = resolve_namepath(namepath)
        self._f = func

    @overload
    def __get__(self, obj: None, cls: type[_T], /) -> Callable[Concatenate[_T, _P], _R_co]: ...
    @overload
    def __get__(self, obj: _T, cls: type[_T] | None = None, /) -> Callable[_P, _R_co]: ...
    def __get__(
        self, obj: _T | None, cls: type[_T] | None = None, /
    ) -> Callable[Concatenate[_T, _P], _R_co] | Callable[_P, _R_co]:
        if obj is None:
            return self._fmeta
        p = functools.partial(self._f.__get__(obj, cls), meta=self.as_ntuple())
        return cast(Callable[_P, _R_co], p)

    def __func__(self) -> Callable[Concatenate[_T, _P], _R_co]:
        return self._f

    def __set_name__(self, obj: Any, name: str) -> None:
        self._n = name
        if obj is None:
            raise ValueError(f"None obj? {obj}")
        if not hasattr(obj, "_infos"):
            setattr(obj, "_infos", {})
        nt = self.as_ntuple()
        obj._infos[self._rnp.namepath] = nt
        # Argument "meta" has incompatible type "AnnotatedMethodInfo"; expected "_P.kwargs"
        p = functools.partial(self._f, meta=nt)  # type: ignore
        self._fmeta = cast(Callable[Concatenate[_T, _P], _R_co], p)

    def as_ntuple(self) -> AnnotatedMethodInfo:
        return AnnotatedMethodInfo(self._rnp, self._n, cast(types.MethodType, self))

    def __repr__(self) -> str:
        # return f"<AnnotatedMethod n: {getattr(self, '_n', 'n/a')} f: {self._f} f_mod: {self._f.__module__} at {id(self):#010x}>"
        return f"<AM n: {getattr(self, '_n', 'n/a')}>"


class rewriter_dec:
    _np: NamePath

    def __init__(self, module: str, qualname: str) -> None:
        self._np = NamePath(module, qualname)

    def __call__(self, func: _F) -> _F:
        return cast(_F, AnnotatedMethod(func, self._np))



# Functions like shrink_types and get_type construct new types at runtime.
# Mypy cannot currently type these functions, so the type signatures for this
# file live in typing.pyi.


def is_list(typ: type) -> bool:
    return is_generic(typ) and name_of_generic(typ) == "List"


def make_typed_dict(*, required_fields=None, optional_fields=None) -> type:
    required_fields = required_fields or {}
    optional_fields = optional_fields or {}
    assert required_fields.keys().isdisjoint(optional_fields.keys())
    return TypedDict(
        DUMMY_TYPED_DICT_NAME,
        {
            "required_fields": TypedDict(
                DUMMY_REQUIRED_TYPED_DICT_NAME, required_fields
            ),
            "optional_fields": TypedDict(
                DUMMY_OPTIONAL_TYPED_DICT_NAME, optional_fields
            ),
        },
    )


def field_annotations(typed_dict) -> Tuple[Dict[str, type], Dict[str, type]]:
    """Return the required and optional fields in the TypedDict."""
    return (
        typed_dict.__annotations__["required_fields"].__annotations__,
        typed_dict.__annotations__["optional_fields"].__annotations__,
    )


def is_anonymous_typed_dict(typ: type) -> bool:
    """Return true if this is an anonymous TypedDict as generated by MonkeyType."""
    return is_typed_dict(typ) and typ.__name__ == DUMMY_TYPED_DICT_NAME


def shrink_typed_dict_types(typed_dicts: List[type], max_typed_dict_size: int) -> type:
    """Shrink a list of TypedDicts into one with the required fields and the optional fields.
    Required fields are keys that appear as a required field in all the TypedDicts.
    Optional fields are those that appear as a required field in only some
    of the TypedDicts or appear as a optional field in even one TypedDict.
    If the same key has multiple value types, then its value is the Union of the value types.
    """
    num_typed_dicts = len(typed_dicts)
    key_value_types_dict = defaultdict(list)
    existing_optional_fields = []
    for typed_dict in typed_dicts:
        required_fields, optional_fields = field_annotations(typed_dict)
        for key, value_type in required_fields.items():
            key_value_types_dict[key].append(value_type)
        existing_optional_fields.extend(optional_fields.items())

    required_fields = {
        key: value_types
        for key, value_types in key_value_types_dict.items()
        if len(value_types) == num_typed_dicts
    }
    optional_fields = defaultdict(list)
    for key, value_types in key_value_types_dict.items():
        if len(value_types) != num_typed_dicts:
            optional_fields[key] = value_types
    for key, value_type in existing_optional_fields:
        optional_fields[key].append(value_type)

    if len(required_fields) + len(optional_fields) > max_typed_dict_size:
        value_type = shrink_types(
            list(
                chain.from_iterable(
                    chain(required_fields.values(), optional_fields.values())
                )
            ),
            max_typed_dict_size,
        )
        return Dict[str, value_type]
    required_fields = {
        key: shrink_types(list(value_types), max_typed_dict_size)
        for key, value_types in required_fields.items()
    }
    optional_fields = {
        key: shrink_types(list(value_types), max_typed_dict_size)
        for key, value_types in optional_fields.items()
    }
    return make_typed_dict(
        required_fields=required_fields, optional_fields=optional_fields
    )


def shrink_types(types, max_typed_dict_size, top: bool = False):
    """Return the smallest type equivalent to Union[types].
    If all the types are anonymous TypedDicts, shrink them ourselves.
    Otherwise, recursively turn the anonymous TypedDicts into Dicts.
    Union will handle deduplicating types (both by equality and subtype relationships).
    """
    types = tuple(types)
    if len(types) == 0:
        return Any
    if all(is_anonymous_typed_dict(typ) for typ in types):
        return shrink_typed_dict_types(types, max_typed_dict_size)
    # Don't rewrite anonymous TypedDict to Dict if the types are all the same,
    # such as [Tuple[TypedDict(...)], Tuple[TypedDict(...)]].
    if all(types_equal(typ, types[0]) for typ in types[1:]):
        return types[0]

    # If they are all lists, shrink their argument types. This way, we avoid
    # rewriting heterogeneous anonymous TypedDicts to Dict.
    if all(is_list(typ) for typ in types):
        annotation = shrink_types(
            (getattr(typ, "__args__")[0] for typ in types), max_typed_dict_size
        )
        return List[annotation]

    all_dict_types = tuple(
        RewriteAnonymousTypedDictToDict(top=top).rewrite(typ, top=top) for typ in types
    )
    return Union[all_dict_types]


def make_iterator(typ):
    return Iterator[typ]


def make_generator(yield_typ, send_typ, return_typ):
    return Generator[yield_typ, send_typ, return_typ]


_BUILTIN_CALLABLE_TYPES = (
    types.FunctionType,
    types.LambdaType,
    types.MethodType,
    types.BuiltinMethodType,
    types.BuiltinFunctionType,
)


def get_dict_type(dct, max_typed_dict_size):
    """Return a TypedDict for `dct` if all the keys are strings.
    Else, default to the union of the keys and of the values."""
    if len(dct) == 0:
        # Special-case this because returning an empty TypedDict is
        # unintuitive, especially when you've "disabled" TypedDict generation
        # by setting `max_typed_dict_size` to 0.
        return Dict[Any, Any]
    if all(isinstance(k, str) for k in dct.keys()) and (
        max_typed_dict_size is None or len(dct) <= max_typed_dict_size
    ):
        return make_typed_dict(
            required_fields={
                k: get_type(v, max_typed_dict_size) for k, v in dct.items()
            }
        )
    else:
        key_type = shrink_types(
            (get_type(k, max_typed_dict_size) for k in dct.keys()), max_typed_dict_size
        )
        val_type = shrink_types(
            (get_type(v, max_typed_dict_size) for v in dct.values()),
            max_typed_dict_size,
        )
        return Dict[key_type, val_type]


def get_type(obj, max_typed_dict_size):
    """Return the static type that would be used in a type hint"""
    if isinstance(obj, type):
        return Type[obj]
    elif isinstance(obj, _BUILTIN_CALLABLE_TYPES):
        return Callable
    elif isinstance(obj, types.GeneratorType):
        return Iterator[Any]
    typ = type(obj)
    if typ is list:
        elem_type = shrink_types(
            (get_type(e, max_typed_dict_size) for e in obj), max_typed_dict_size
        )
        return List[elem_type]
    elif typ is set:
        elem_type = shrink_types(
            (get_type(e, max_typed_dict_size) for e in obj), max_typed_dict_size
        )
        return Set[elem_type]
    elif typ is dict:
        return get_dict_type(obj, max_typed_dict_size)
    elif typ is defaultdict:
        key_type = shrink_types(
            (get_type(k, max_typed_dict_size) for k in obj.keys()), max_typed_dict_size
        )
        val_type = shrink_types(
            (get_type(v, max_typed_dict_size) for v in obj.values()),
            max_typed_dict_size,
        )
        return DefaultDict[key_type, val_type]
    elif typ is tuple:
        return Tuple[tuple(get_type(e, max_typed_dict_size) for e in obj)]
    return typ


NoneType = type(None)
NotImplementedType = type(NotImplemented)
mappingproxy = type(range.__dict__)


T = TypeVar("T")


class GenericTypeRewriter(Generic[T], ABC):
    _infos: dict[NamePath, AnnotatedMethodInfo]
    _infos_ro: types.MappingProxyType[NamePath, AnnotatedMethodInfo]
    _top: bool

    def __init__(self, top: bool = False) -> None:
        super().__init__()
        self._top = top
        if not hasattr(self, "_infos"):
            self._infos = {}
        self._infos_ro = types.MappingProxyType(self._infos)

    @property
    def registry(self) -> types.MappingProxyType[NamePath, AnnotatedMethodInfo]:
        return self._infos_ro

    @property
    def top(self) -> bool:
        return self._top

    def _call_annotated_method(
        self, method_info: AnnotatedMethodInfo, /, *args: Any, **kwargs: Any
    ) -> Any:
        m = method_info.method.__get__(self, type(self))  # type: ignore
        return m(*args, **kwargs)

    @abstractmethod
    def make_builtin_tuple(self, elements): ...

    @abstractmethod
    def make_container_type(self, container_type, element): ...

    @abstractmethod
    def make_anonymous_typed_dict(self, required_fields, optional_fields): ...

    @abstractmethod
    def make_builtin_typed_dict(self, name, annotations, total): ...

    @abstractmethod
    def generic_rewrite(self, typ, caller: str | None = None): ...

    @abstractmethod
    def rewrite_container_type(self, container_type): ...

    @abstractmethod
    def rewrite_malformed_container(self, container):
        raise RuntimeError("rewrite_malformed_container ABC is banned")

    @abstractmethod
    def rewrite_type_variable(self, type_variable): ...

    def _rewrite_container(self, cls, container):
        if container.__module__ != "typing":
            print(f"_rewrite_container() container: {container} mod: {container.__module__}")
            return self.rewrite_malformed_container(container)
        args = getattr(container, "__args__", None)
        if args is None:
            return self.rewrite_malformed_container(container)
        elif args == ((),):  # special case of empty tuple `Tuple[()]`
            print(f"this better be a tuple: cls: {cls} container: {container}")
            elems = self.make_builtin_tuple(())
        else:
            elems = self.make_builtin_tuple(
                self.rewrite(elem) for elem in container.__args__
            )
        return self.make_container_type(self.rewrite_container_type(cls), elems)

    @rewriter_dec("typing", "Dict")
    def rewrite_Dict(self, dct, meta: AMI = AMIS):
        return self._rewrite_container(Dict, dct)

    @rewriter_dec("typing", "List")
    def rewrite_List(self, lst, meta: AMI = AMIS):
        return self._rewrite_container(List, lst)

    @rewriter_dec("typing", "Set")
    def rewrite_Set(self, st, meta: AMI = AMIS):
        return self._rewrite_container(Set, st)

    @rewriter_dec("typing", "Tuple")
    def rewrite_Tuple(self, tup, meta: AMI = AMIS):
        return self._rewrite_container(Tuple, tup)

    @rewriter_dec("typing", "Generator")
    def rewrite_Generator(self, generator, meta: AMI = AMIS):
        print(f"GTR(): rewrite_Generator: generator: {generator}")
        return self._rewrite_container(Generator, generator)

    def rewrite_anonymous_TypedDict(self, typed_dict):
        assert is_anonymous_typed_dict(typed_dict)
        required_fields, optional_fields = field_annotations(typed_dict)
        return self.make_anonymous_typed_dict(
            required_fields={
                name: self.rewrite(typ) for name, typ in required_fields.items()
            },
            optional_fields={
                name: self.rewrite(typ) for name, typ in optional_fields.items()
            },
        )

    @rewriter_dec("typing_extensions", "TypedDict")
    def rewrite_TypedDict(self, typed_dict, meta: AMI = AMIS):
        if is_anonymous_typed_dict(typed_dict):
            return self.rewrite_anonymous_TypedDict(typed_dict)
        return self.make_builtin_typed_dict(
            typed_dict.__name__,
            {
                name: self.rewrite(typ)
                for name, typ in typed_dict.__annotations__.items()
            },
            total=typed_dict.__total__,
        )

    @rewriter_dec("typing", "Union")
    def rewrite_Union(self, union, meta: AMI = AMIS) -> Any:
        print(f"GenericTypeRewriter.rewrite_Union() self: {self} union: {union} meta: {meta}")
        return self._rewrite_container(Union, union)

    def rewrite(self, typ, caller: str | None = None, top: bool = False):
        cstr = caller if caller is not None else ""
        print(f"GTR({cstr}).rw() typ: {typ}")
        print(f"GTR() registry: id: {id(self.registry):#010x} reg: {self.registry}")
        callstr = f"GTR({cstr}).rw()"
        r = None
        if is_any(typ):
            np = get_namepath(Any)
        elif is_union(typ):
            np = get_namepath(Union)
        elif is_typed_dict(typ):
            np = get_namepath(TypedDict)
        elif is_generic(typ):
            np = get_namepath(typ)
        else:
            # raise TypeError(f"Unknown type: {typ}")
            r = self.generic_rewrite(typ)
            print(f"rewrite({typ}) generic2 => {r}")
            return r
        rewriter = self.registry.get(np)
        if rewriter:
            print(f"GTR({cstr}).rw() rewriter: {rewriter}")
            r = self._call_annotated_method(rewriter, typ)
            print(f"GTR({cstr}).rw() typ: {typ} decorator => {r}")
            return r
        if isinstance(typ, TypeVar):
            r = self.rewrite_type_variable(typ)
            print(f"GTR({cstr}).rw() typ: {typ} typevar => {r}")
            return r
        r = self.generic_rewrite(typ, caller=callstr)
        print(f"GTR({cstr}).rw() typ: {typ} generic => {r}")
        return r


class TypeRewriter(GenericTypeRewriter[type]):
    """TypeRewriter provides a visitor for rewriting parts of types"""

    def make_anonymous_typed_dict(self, required_fields, optional_fields):
        return make_typed_dict(
            required_fields=required_fields, optional_fields=optional_fields
        )

    def make_builtin_typed_dict(self, name, annotations, total):
        return TypedDict(name, annotations, total=total)

    def generic_rewrite(self, typ, caller: str | None = None):
        return typ

    def rewrite_container_type(self, container_type):
        return container_type

    def rewrite_malformed_container(self, container):
        raise RuntimeError("rewrite_malformed_container is banned")
        return container

    def rewrite_type_variable(self, type_variable):
        return type_variable

    def make_builtin_tuple(self, elements):
        return tuple(elements)

    def make_container_type(self, container_type, element):
        return container_type[element]


class RemoveEmptyContainers(TypeRewriter):
    """Remove redundant, empty containers from union types.

    Empty containers are typed as C[Any] by MonkeyType. They should be removed
    if there is a single concrete, non-null type in the Union. For example,

        Union[Set[Any], Set[int]] -> Set[int]

    Union[] handles the case where there is only a single type left after
    removing the empty container.
    """

    def _is_empty(self, typ):
        args = getattr(typ, "__args__", [])
        return args and all(is_any(e) for e in args)

    @rewriter_dec("typing", "Union")
    def rewrite_Union(self, union, meta: AMI = AMIS):
        print(f"RemoveEmptyContainers.rewrite_Union() self: {self} union: {union} meta: {meta}")
        elems = tuple(self.rewrite(e) for e in union.__args__ if not self._is_empty(e))
        if elems:
            return Union[elems]
        return union


class RewriteConfigDict(TypeRewriter):
    """Union[Dict[K, V1], ..., Dict[K, VN]] -> Dict[K, Union[V1, ..., VN]]"""


    @rewriter_dec("typing", "Union")
    def rewrite_Union(self, union, meta: AMI = AMIS):
        print(f"RewriteConfigDict.rewrite_Union() self: {self} union: {union} meta: {meta}")
        key_type = None
        value_types = []
        for e in union.__args__:
            if not is_generic_of(e, Dict):
                return union
            key_type = key_type or e.__args__[0]
            if key_type != e.__args__[0]:
                return union
            value_types.extend(e.__args__[1:])
        return Dict[key_type, Union[tuple(value_types)]]


class RewriteLargeUnion(TypeRewriter):
    """Rewrite Union[T1, ..., TN] as Any for large N."""

    def __init__(self, max_union_len: int = 5, top: bool = False):
        super().__init__(top=top)
        self.max_union_len = max_union_len

    def _rewrite_to_tuple(self, union):
        """Union[Tuple[V, ..., V], Tuple[V, ..., V], ...] -> Tuple[V, ...]"""
        value_type = None
        for t in union.__args__:
            if not is_generic_of(t, Tuple):
                return None
            value_type = value_type or t.__args__[0]
            if not all(vt is value_type for vt in t.__args__):
                return None
        return Tuple[value_type, ...]

    @rewriter_dec("typing", "Union")
    def rewrite_Union(self, union, meta: AMI = AMIS):
        print(f"RewriteLargeUnion.rewrite_Union() self: {self} union: {union} meta: {meta}")
        if len(union.__args__) <= self.max_union_len:
            return union

        rw_union = self._rewrite_to_tuple(union)
        if rw_union is not None:
            return rw_union

        try:
            for ancestor in inspect.getmro(union.__args__[0]):
                if ancestor is not object and all(
                    issubclass(t, ancestor) for t in union.__args__
                ):
                    return ancestor
        except (TypeError, AttributeError):
            pass
        return Any


class RewriteAnonymousTypedDictToDict(TypeRewriter):
    """TypedDict('Foo', {"k": v1, ...}) -> Dict[str, Union[v1, ...]]."""

    def rewrite_anonymous_TypedDict(self, typed_dict):
        assert is_anonymous_typed_dict(typed_dict)
        required_fields, optional_fields = field_annotations(typed_dict)
        all_value_types = [*required_fields.values(), *optional_fields.values()]
        if not all_value_types:
            # Special-case this because we can't justify any type.
            return Dict[Any, Any]
        return Dict[str, Union[tuple(self.rewrite(typ) for typ in all_value_types)]]


class ChainedRewriter(TypeRewriter):
    def __init__(self, rewriters: Iterable[TypeRewriter], top: bool = False) -> None:
        super().__init__(top=top)
        self.rewriters = rewriters

    def rewrite(self, typ, caller: str | None = None, top: bool = False):
        cstr = caller if caller is not None else ""
        print(f"CHN({cstr}).rw() typ: {typ}")
        for i, rw in enumerate(self.rewriters):
            print(f"CHN({cstr}).rw() rw[{i}] typ: {typ}")
            callstr = f"CHN({cstr})[{i}].rw()"
            typ = rw.rewrite(typ, caller=callstr, top=top)
        return typ


class NoOpRewriter(TypeRewriter):
    def rewrite(self, typ, caller: str | None = None, top: bool = False):
        cstr = caller if caller is not None else ""
        print(f"NOP({cstr}).rw() typ: {typ}")
        return typ


class RewriteGenerator(TypeRewriter):
    """Returns an Iterator, if the send_type and return_type of a Generator is None"""

    @rewriter_dec("typing", "Generator")
    def rewrite_Generator(self, generator, meta: AMI = AMIS):
        print(f"RG(): rewrite_Generator: generator: {generator}")
        print(f"RG() registry: id: {id(self.registry):#010x} reg: {self.registry}")
        args = generator.__args__
        if args[1] is NoneType and args[2] is NoneType:
            return Iterator[args[0]]
        return generator


class RewriteMostSpecificCommonBase(TypeRewriter):
    """
    Relace a union of classes by the most specific
    common base of its members (while avoiding multiple
    inheritance), i.e.,

    Union[Derived1, Derived2] -> Base
    """

    def _compute_bases(self, klass):
        """
        Return list of bases of a given class,
        going from general (i.e., closer to object)
        to specific (i.e., closer to class).
        The list ends with the class itself, its
        first element is the most general base of
        the class up to (but excluding) any
        base class having multiple inheritance
        or the object class itself.
        """
        bases = []

        curr_klass = klass

        while curr_klass is not object:
            bases.append(curr_klass)

            curr_bases = curr_klass.__bases__

            if len(curr_bases) != 1:
                break

            curr_klass = curr_bases[0]
        return bases[::-1]

    def _merge_common_bases(self, first_bases, second_bases):
        """
        Return list of bases common to both* classes,
        going from general (i.e., closer to object)
        to specific (i.e., closer to both classes).
        """
        merged_bases = []

        # Only process up to shorter of the lists
        for first_base, second_base in zip(first_bases, second_bases):
            if first_base is second_base:
                merged_bases.append(second_base)
            else:
                break

        return merged_bases

    @rewriter_dec("typing", "Union")
    def rewrite_Union(self, union, meta: AMI = AMIS):
        print(f"RewriteMostSpecificCommonBase.rewrite_Union() self: {self} union: {union} meta: {meta}")
        """
        Rewrite the union if possible, if no meaningful rewrite is possible,
        return the original union.
        """
        klasses = union.__args__

        all_bases = []

        for klass in klasses:
            klass_bases = self._compute_bases(klass)
            all_bases.append(klass_bases)

        common_bases = functools.reduce(self._merge_common_bases, all_bases)

        if common_bases:
            return common_bases[-1]
        return union


DEFAULT_REWRITER = ChainedRewriter(
    (
        RemoveEmptyContainers(top=True),
        RewriteConfigDict(top=True),
        RewriteLargeUnion(top=True),
        RewriteGenerator(top=True),
    ),
    top=True,
)
