from typing import Generator, Sequence
import decimal
from functools import wraps
import inspect
import logging
import math
import os
import sys
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from collections import OrderedDict
from tools.util.path_tools import relpath
import torch
from decimal import Decimal
import numpy as np
from tools.error import NoIterationTypeError, NoSimpleTypeError, ArgumentNoneError
import hashlib
from tools.util.progress_factory import ProgressFactory
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE, _DEFAULT, DEFAULT
from typing import Callable, Tuple
from tools.util.reflection import class_name
from tools.util.format import _custom_traceback
from tools.logger.logging import tools_logger as logger
from traceback import FrameSummary, extract_stack
from tools.transforms.to_tensor import tensorify
from tools.transforms.to_tensor_image import tensorify_image
import gc
from torch.nn import ModuleList
from tools.util.sized_generator import sized_generator, SizedGenerator


def set_jit_enabled(enabled: bool):
    """Enables or disables JIT.

    Parameters
    ----------
    enabled : bool
        If JIT should be enabled.
    """
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()


def jit_enabled():
    """Gets the current JIT state."""
    if torch.__version__ < "1.7":
        return torch.jit._enabled
    else:
        return torch.jit._state._enabled.enabled


def get_weight_normalized_param_groups(network: torch.nn.Module,
                                       weight_decay: float,
                                       norm_suffix: str = 'weight_g',
                                       name_prefix: str = ''):

    norm_params = []
    unnorm_params = []
    for n, p in network.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    prefix = (name_prefix.strip() +
              (" " if len(name_prefix.strip()) > 0 else ""))
    param_groups = [{'name': f'{prefix}normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': f'{prefix}unnormalized', 'params': unnorm_params}]
    return param_groups


def count_parameters(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Counts the number of parameters in the given model.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch module to count the parameters of.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing the name, number of learnable parameters and id of the parameter.
        Name is the "path" to the parameter in the model.
    """
    param_list = []
    total_params = 0
    for i, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        param_list.append(dict(name=name, learnable_params=params, id=i))
    param_list.append(
        dict(name="total", learnable_params=total_params, id=len(param_list)))
    return param_list


def as_tensors(keep_output: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for converting all input arguments to tensors.

    Will convert all input arguments numpy arrays to tensors, if they are not already.

    Parameters
    ----------
    keep_output : bool, optional
        If the output should be keept as it is, by default False

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        The decorator.
    """
    from tools.util.numpy import numpyify

    def decorator(fnc: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs) -> Any:
            nonlocal keep_output
            keep_output = kwargs.pop("keep_tensor", keep_output)
            is_numpy = False

            def process(v):
                nonlocal is_numpy
                if isinstance(v, np.ndarray):
                    is_numpy = True
                    return tensorify(v)
                if isinstance(v, list) or isinstance(v, tuple):
                    return tensorify(v)
                return v
            args = [process(arg) for arg in args]
            kwargs = {k: process(v) for k, v in kwargs.items()}
            ret = fnc(*args, **kwargs)
            if keep_output or not is_numpy:
                return ret
            return numpyify(ret)
        return wrapper
    return decorator


def fourier(x: torch.Tensor) -> torch.Tensor:
    """2D fourier transform with normalization and shift.

    Parameters
    ----------
    x : torch.Tensor
        Spatial data to transform.

    Returns
    -------
    torch.Tensor
        Complex fourier spectrum.
    """
    return torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))


def inverse_fourier(x: torch.Tensor) -> torch.Tensor:
    """2D inverse fourier transform with normalization and shift.

    Parameters
    ----------
    x : torch.Tensor
        Fourier data to transform.

    Returns
    -------
    torch.Tensor
        Spatial output.
    """
    return torch.fft.ifft2(torch.fft.ifftshift(x), norm='forward')


def complex_dtype(float_dtype: torch.dtype = torch.float32) -> torch.dtype:
    """Returns the corresponding precision complex dtype for a given float dtype.

    Parameters
    ----------
    float_dtype : torch.dtype, optional
        Float dtype to convert, by default torch.float32

    Returns
    -------
    torch.dtype
        The complex equivalent
    """
    if float_dtype == torch.float64:
        return torch.complex128
    elif float_dtype == torch.float32:
        return torch.complex64
    elif float_dtype == torch.float16:
        return torch.complex32
    else:
        raise ValueError(f"Float dtype {float_dtype} is not supported.")


def numpy_to_torch_dtype(dtype: Union[str, np.dtype]) -> torch.dtype:
    """Converts a numpy dtype to a torch dtype.

    Parameters
    ----------
    dtype : Union[str, np.dtype]
        Numpy dtype to convert or string representation.

    Returns
    -------
    torch.dtype
        The corresponding torch dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported.
    """
    numpy_to_torch_dtype_dict = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    if dtype in numpy_to_torch_dtype_dict:
        return numpy_to_torch_dtype_dict[dtype]
    elif isinstance(dtype, np.dtype):
        try:
            dt = getattr(np, dtype.name)
            return numpy_to_torch_dtype(dt)
        except AttributeError:
            raise ValueError(f"Unsupported numpy dtype {dtype.name}")
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def torch_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    """Takes a torch dtype and returns the corresponding numpy dtype.

    Parameters
    ----------
    dtype : torch.dtype
        The torch dtype to convert.

    Returns
    -------
    np.dtype
        Numpy dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported.
    """
    torch_to_numpy_dtype_dict = {
        torch.bool: bool,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128
    }
    if dtype in torch_to_numpy_dtype_dict:
        return torch_to_numpy_dtype_dict[dtype]
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def detach_module(module: torch.nn.Module) -> torch.nn.Module:
    """Detaches a module from the computation graph.

    Parameters
    ----------
    module : torch.nn.Module
        Module to detach.

    Returns
    -------
    torch.nn.Module
        Detached module.
    """

    for param in module.parameters():
        param.requires_grad = False
    return module


def get_index_tuple(shape: tuple, slice_or_indexer: Any, dim: int) -> tuple:
    idx = [slice(None) for _ in range(len(shape))]
    idx[dim] = slice_or_indexer
    return tuple(idx)


def batched_generator_exec(
        batched_params: List[Union[str, dict]],
        default_batch_size: int = 1,
        default_multiprocessing: bool = False,
        default_num_workers: int = 4,
        default_explicit_garbage_collection: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., SizedGenerator]]:
    """
    Decorator for executing a function in batches.
    The wrapped function will be executed in batches based on the batched_params.

    Example:
    ---------

    >>> @batched_generator_exec(['t', dict(name='other_arg', batch_dim=1)], default_batch_size=3)
        def some_op(resolution: tuple, t: torch.Tensor, other_arg: torch.Tensor) -> torch.Tensor:
            return torch.rand(*resolution).unsqueeze(0).repeat(len(t), 1, 1)

    >>> my_ts = torch.rand(10, 5)
        other_arg = torch.rand(9, 10)
        resolution = (10, 20)
        combined_result = torch.cat((list(some_op(resolution, t=my_ts, batch_size=3, other_arg=other_arg))), dim=0).shape


    Parameters
    ----------
    batched_params : List[Union[str, dict]]
        List of parameters that should be batched. Each parameter can be a string or a dictionary.
        If a string is given, the parameter will be batched with the default batch dimension.
        If a dictionary is given, it must contain the key 'name' with the parameter name and optionally 'batch_dim' with the batch dimension.

    default_batch_size : int, optional
        Default batch size to use, by default 1

    default_multiprocessing : bool, optional
        If multiprocessing should be used, by default False

    default_num_workers : int, optional
        Number of workers to use for multiprocessing, by default 4

    default_explicit_garbage_collection : bool, optional
        If explicit garbage collection should be used after each batching, by default False
        Will call torch.cuda.empty_cache() and gc.collect()
    Returns
    -------
    callable
        The decorator.

    """
    def _define_batched_params(
            param: Union[str, dict],
            default_batch_dim: int = 0,
    ) -> dict:
        param_dict = dict()
        if isinstance(param, str):
            param_dict['name'] = param
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise ValueError(
                f"param must be either str or dict, got {type(param)}")
        if 'name' not in param_dict:
            raise ValueError("param must contain 'name' key")
        if 'batch_dim' not in param_dict:
            param_dict['batch_dim'] = default_batch_dim
        return param_dict

    def map_args_to_kwargs(
        list_args: List,
        function_params: Dict[str, inspect.Parameter],
    ) -> Dict[str, Any]:
        if len(list_args) == 0:
            return dict()
        return dict(zip(function_params.keys(), list_args))

    def get_batch_size(
            value: Any,
            param_spec: dict) -> int:
        if isinstance(value, (list, tuple)):
            if param_spec['batch_dim'] != 0:
                raise ValueError(
                    f"For list or tuple, batch_dim must be 0, got {param_spec['batch_dim']}")
            return len(value)
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f"Expected torch.Tensor or np.ndarray, got {type(value)} for parameter {param_spec['name']}")
        if param_spec['batch_dim'] >= len(value.shape):
            raise ValueError(
                f"Batch dimension {param_spec['batch_dim']} is out of bounds for tensor of shape {value.shape}")
        return value.shape[param_spec['batch_dim']]

    def slice_value(value: Any, s: slice, batch_dim: int) -> Any:
        if isinstance(value, (torch.Tensor, np.ndarray)):
            idx = get_index_tuple(value.shape, s, batch_dim)
            return value[idx]
        if isinstance(value, (list, tuple)):
            return value[s]
        raise ValueError(f"Unsupported type {type(value)}")

    def assembled_batched_params(batched_params: List[dict],
                                 args: List,
                                 kwargs: Dict[str, Any],
                                 function_params: Dict[str, inspect.Parameter],
                                 batch_size: int = 1
                                 ) -> List[Dict[str, Any]]:
        ret_kwargs = list()
        input_args = map_args_to_kwargs(args, function_params)
        # Check if overlapping keys
        intersection = set(input_args.keys()) & set(kwargs.keys())
        if len(intersection) > 0:
            raise ValueError(
                f"Multiple values for argument(s): {list(intersection)}")
        combined_kwargs = {**input_args, **kwargs}
        batch_sizes = {x: get_batch_size(
            v, batched_params[x]) for x, v in combined_kwargs.items() if x in batched_params}
        # Check if batch sizes are the same
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(
                f"Batch dimension size are not the same: {batch_sizes}.")
        batch_dimension_size = list(batch_sizes.values())[0]
        num_batches = math.ceil(batch_dimension_size / batch_size)
        slices = [slice(i * batch_size, min((i + 1) * batch_size,
                        batch_dimension_size)) for i in range(num_batches)]
        ret_kwargs = [dict() for _ in range(num_batches)]

        for k, v in combined_kwargs.items():
            if k in batched_params:
                for i, s in enumerate(slices):
                    ret_kwargs[i][k] = slice_value(
                        v, s, batched_params[k]['batch_dim'])
            else:
                for i in range(num_batches):
                    ret_kwargs[i][k] = v
        return ret_kwargs

    def garbage_collect():
        torch.cuda.empty_cache()
        gc.collect()

    _batched_params: Dict[str, Dict[str, Any]] = dict()
    for p in batched_params:
        def_p = _define_batched_params(p)
        _batched_params[def_p['name']] = def_p

    def decorator(function: callable) -> callable:
        function_params = dict(inspect.signature(function).parameters)

        @wraps(function)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal _batched_params
            mod_kwargs = kwargs.copy()
            batch_size = mod_kwargs.pop('batch_size', default_batch_size)
            explicit_gc = mod_kwargs.pop(
                'explicit_garbage_collection', default_explicit_garbage_collection)
            use_multiprocessing = mod_kwargs.pop(
                'multiprocessing', default_multiprocessing)

            if use_multiprocessing:
                try:
                    import pathos  # type: ignore
                except (ImportError, ModuleNotFoundError) as e:
                    logger.warning(
                        "Multiprocessing is enabled, but pathos is not available. Disabling multiprocessing.")
                    use_multiprocessing = False

            num_workers = mod_kwargs.pop('num_workers', default_num_workers)

            batched_kwargs = assembled_batched_params(
                _batched_params, args=args, kwargs=mod_kwargs, function_params=function_params, batch_size=batch_size)

            if not use_multiprocessing:
                # Yield size
                yield len(batched_kwargs)
                for batch_idx, batch_kwargs in enumerate(batched_kwargs):
                    yield function(**batch_kwargs)
                    if explicit_gc:
                        garbage_collect()

            else:
                from pathos.pools import ProcessPool  # type: ignore

                listed_args = [list((bkw[k] for bkw in batched_kwargs))
                               for k in function_params if k in batched_kwargs[0]]

                # Yield size
                yield len(batched_kwargs)
                with ProcessPool(num_workers) as p:
                    results = p.imap(function, *listed_args)
                    for r in results:
                        yield r
                        if explicit_gc:
                            garbage_collect()
        wrap_gen = sized_generator()(wrapper)
        return wrap_gen
    return decorator


def batched_exec(*input,
                 func: Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
                 batch_size: int,
                 progress_bar: bool = False,
                 pf: Optional[ProgressFactory] = None,
                 free_memory: bool = False,
                 progress_bar_delay: float = 2.,
                 **kwargs
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Execute a function in batches.

    Parameters
    ----------
    input : List[torch.Tensor]
        Should be a list of tensors which have a common batch dimension.
        (B, ...) this is the first dimension and will be used to batch the execution.

    func : Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        Function to execute. Should take a tensors as input and return a tensor as output, or a tuple of tensors.

    batch_size : int
        The batch size to use for the execution.

    progress_bar : bool, optional
        If a progress bar should be displayed, by default False

    pf : Optional[ProgressFactory], optional
        Progress factory to reuse the progress bar, by default None

    free_memory : bool, optional
        If memory should be freed after each batch, by default False
        Will call torch.cuda.empty_cache() and gc.collect()

    progress_bar_delay : float, optional
        Delay in seconds before which needs to expire before a bar is displayed, by default 2.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        Concatenated results of the function execution.
        Returns a single tensor if the function returns a single tensor,
        otherwise returns a tuple of tensors.
        The order of the output tensors corresponds to the order of the outputs from the function.

        Size of the output tensors will be (B, ...) where B is the batch size of the input tensors.
    """
    import gc
    results = dict()
    num_outputs = None
    is_tuple = False

    if progress_bar:
        if pf is None:
            pf = ProgressFactory()

    if len(input) == 0:
        raise ValueError("At least one input tensor is required")
    B = input[0].shape[0]
    full_batches, remainder = divmod(B, batch_size)

    # Compute slices
    slices = []

    for i in range(0, full_batches):
        slices.append(slice(i*batch_size, (i+1)*batch_size))
    if remainder > 0:
        slices.append(slice(full_batches*batch_size, B))

    bar = None
    if progress_bar:
        bar = pf.tqdm(total=len(
            slices), desc=f"Batched Execution: {class_name(func)}", tag=f"BATCH_EXEC_" + class_name(func), is_reusable=True, delay=progress_bar_delay)

    for s in slices:
        _ins = [i[s] for i in input]
        r = func(*_ins, **kwargs)
        if isinstance(r, tuple):
            if num_outputs is None:
                num_outputs = len(r)
                is_tuple = True
        else:
            if num_outputs is None:
                num_outputs = 1
            r = (r,)
        for i in range(num_outputs):
            if i not in results:
                results[i] = []
            results[i].append(r[i])

        if free_memory:
            gc.collect()
            torch.cuda.empty_cache()
        if progress_bar:
            bar.update(1)
    outs = tuple(torch.cat(results[i], dim=0) for i in range(num_outputs))
    if not is_tuple:
        return outs[0]
    return outs


def index_of_first(values: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
    """Searches for the index of the first occurence of the search tensor in the values tensor.

    Tested for 1D tensors. Returns -1 if the search tensor is not found in the values tensor.

    Example:
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    search = torch.tensor([5, 6, 7, 12])

    index_of_first(values, search) -> torch.tensor([4, 5, 6, -1])

    Parameters
    ----------
    values : torch.Tensor
        Values tensor to search in.
    search : torch.Tensor
        Search tensor to search for.

    Returns
    -------
    torch.Tensor
        Index tensor of the first occurence of the search tensor in the values tensor.
    """
    values = tensorify(values)
    search = tensorify(search)
    is_scalar = False
    E = tuple(values.shape)

    if len(search.shape) == 0:
        search = search.unsqueeze(0)
        is_scalar = True

    S = tuple(search.shape)
    ER = tuple(torch.ones(len(E), device=values.device, dtype=torch.int))
    ES = tuple(torch.ones(len(S), device=values.device, dtype=torch.int))
    res = values[..., None].repeat(
        *ER, *S) == search[None, ...].repeat(*E, *ES)
    out = torch.zeros(S, dtype=torch.int)
    out.fill_(-1)
    aw = torch.argwhere(res)
    search_found, where_inverse = torch.unique(aw[:, -1], return_inverse=True)
    for i, s in enumerate(search_found):
        widx = aw[where_inverse == i][0]  # Select first match
        out[s] = widx[0]

    if is_scalar:
        return out.squeeze(0)
    return out


def angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Calculates the clockwise angle between two vectors.

    Preserves the sign of the angle based on the cross product of the two vectors.

    Parameters
    ----------
    v1 : torch.Tensor
        The first vector. Shape: (..., 2) or (..., 3)

    v2 : torch.Tensor
        The second vector. Shape: (..., 2) or (..., 3)

    Returns
    -------
    torch.Tensor
        The angle between the two vectors. Shape: (..., 1) or (..., 2)
    """

    v1, shp = flatten_batch_dims(v1, -2)
    v2 = flatten_batch_dims(v2, -2)[0]

    # Normalize the vectors
    nVs = v1 / torch.norm(v1, dim=-1, keepdim=True)
    mV = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Calculate the cross product (z-component)
    if nVs.shape[-1] == 2:
        cross_product = nVs[:, 0] * mV[:, 1] - nVs[:, 1] * mV[:, 0]
    elif nVs.shape[-1] == 3:
        cross_product = torch.cross(nVs, mV, dim=-1)
    else:
        raise ValueError("Vectors must be 2D or 3D")

    # Calculate the dot product
    dot_product = torch.sum(nVs * mV, dim=-1)

    # Use atan2 to calculate the angle, handling the sign based on the cross product
    angle = torch.atan2(cross_product, dot_product)

    # Adjust the angle to the desired range (0 to 2pi)
    angle = (angle + 2 * torch.pi) % (2 * torch.pi)

    return unflatten_batch_dims(angle, shp)


def complex_sign_angle(v: torch.Tensor) -> torch.Tensor:
    """Calculates the angle of a complex vector and preserves the sign of the angle.
    Angle is in the range of -pi to pi.

    Parameters
    ----------
    v : torch.Tensor
        Complex vector. Shape: (...)

    Returns
    -------
    torch.Tensor
        The angle of the complex vector. Shape: (...)
    """
    if "complex" not in str(v.dtype):
        raise ValueError("Input must be a complex tensor.")


def grad_at(
    fnc: Callable[[torch.Tensor], torch.Tensor],
    x: VEC_TYPE,
    create_graph: bool = True,
    retain_graph: bool = False,
) -> torch.Tensor:
    """
    Calculates the first order gradient of a function at a given point.
    Quick helper function to calculate arbitrary gradients of pytorch functions.

    Parameters
    ----------
    fnc : Callable[[torch.Tensor], torch.Tensor]
        Function to calculate the gradient of.

    x : torch.Tensor
        Point to calculate the gradient at.

    create_graph : bool, optional
        If the graph should be created, by default True

    retain_graph : bool, optional
        If the graph should be retained, by default False

    Returns
    -------
    torch.Tensor
        The gradient of the function at the given point.
    """
    x = tensorify(x)
    with torch.set_grad_enabled(True):
        x = x.requires_grad_(True)
        grd = torch.autograd.grad(
            fnc(x).sum(), x, create_graph=create_graph, retain_graph=retain_graph)[0]
        return grd


class TensorUtil():
    """Static class for using complex tensor calculations and tensor related utilities"""

    @staticmethod
    def to(object: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Any:
        """Takes an arbitrary object an tries to move all internal tensors to a device and specified dtype.

        Parameters
        ----------
        object : Any
            The object to query.
        dtype : Optional[torch.dtype], optional
            The dtype where tensors should be converted into, None makes no conversion., by default None
        device : Optional[torch.device], optional
            The device where tensors should be moved to, by default None

        Returns
        -------
        Any
            The object with changed properties.

        Raises
        ------
        ArgumentNoneError
            If object is None
        """
        if object is None:
            raise ArgumentNoneError("object")
        if dtype is None and device is None:
            # Do nothing because no operation is specified.
            return object
        return TensorUtil._process_value(object, "", object, dtype=dtype, device=device)

    @staticmethod
    def to_hash(object: Any) -> Any:
        """Takes an object graph and hashes all tensors with sha256.
        Keeps the object graph structure, but replaces all tensors with their sha256 hash.

        Parameters
        ----------
        object : Any
            An arbitrary object graph containing tensors.

        Returns
        -------
        Any
            Object graph with tensors replaced by their sha256 hash.
        """
        def _tensor_hash(x: torch.Tensor) -> str:
            return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        return TensorUtil.apply_deep(object, fnc=_tensor_hash)

    @staticmethod
    def reset_parameters(object: Any, memo: Set[Any] = None) -> None:
        """Resets all parameters of the given object and its children,
        by calling reset_parameters() on supported objects.

        Parameters
        ----------
        object : Any
            The object / module to reset.
        memo : Set[Any], optional
            Memo for already visited objects, by default None
        """
        if memo is None:
            memo = set()
        try:
            if hasattr(object, "__hash__") and object.__hash__ is not None:
                if object in memo:
                    return
                else:
                    memo.add(object)
        except TypeError as err:
            # Unhashable type
            pass

        if hasattr(object, "reset_parameters"):
            ret = object.reset_parameters()
            if ret is not None and isinstance(ret, bool) and ret:
                # If reset_parameters returns True, it means that it has reset its child parameters
                return ret

        # Recursively reset all parameters, if possible
        if isinstance(object, torch.nn.Module):
            # Proceed with child modules
            for m in object.modules():
                TensorUtil.reset_parameters(m, memo=memo)
        return True

    @staticmethod
    def _process_value(value: Any,
                       name: str,
                       context: Dict[str, Any],
                       dtype: Optional[torch.dtype] = None,
                       device: Optional[torch.device] = None) -> Any:
        try:
            return TensorUtil._process_simple_type(value, name, context, dtype=dtype, device=device)
        except NoSimpleTypeError:
            try:
                return TensorUtil._process_iterable(value, name, context, dtype=dtype, device=device)
            except NoIterationTypeError:
                return value

    @staticmethod
    def _process_simple_type(value, name: str,
                             context: Dict[str, Any],
                             dtype: Optional[torch.dtype] = None,
                             device: Optional[torch.device] = None) -> Any:
        if value is None:
            return value
        elif isinstance(value, torch.Tensor):
            return TensorUtil._process_tensor(value, name, context, dtype=dtype, device=device)
        elif hasattr(value, "to"):
            try:
                return TensorUtil._process_tensor(value, name, context, dtype=dtype, device=device)
            except (TypeError) as err:
                # Propably wrong argument combination
                logging.warning(
                    f"Type Error on invoking 'to' of value: {value}. \n {err}")
                raise NoSimpleTypeError()
        elif isinstance(value, (int, float, str, decimal.Decimal)):
            return value
        else:
            raise NoSimpleTypeError()

    @staticmethod
    def _process_tensor(value: torch.Tensor, name: str,
                        context: Dict[str, Any],
                        dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None
                        ) -> Any:
        return value.to(dtype=dtype, device=device)

    @staticmethod
    def _process_dict(value: Dict[str, Any],
                      name: str,
                      context: Dict[str, Any],
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None) -> Dict[str, Any]:
        # Works same as json hook, creates objects from inside to outside
        ret = {}
        # Handling internals
        for k, v in value.items():
            ret[k] = TensorUtil._process_value(
                v, name, context, dtype=dtype, device=device)
        # Converting ret with hook if childrens are processed
        return ret

    @staticmethod
    def _process_iterable(value, name: str,
                          context: Dict[str, Any],
                          dtype: Optional[torch.dtype] = None,
                          device: Optional[torch.device] = None) -> Any:
        if isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            a = []
            for subval in value:
                a.append(TensorUtil._process_value(
                    subval, name, value, dtype=dtype, device=device))
            if isinstance(value, list):
                return a
            else:
                return tuple(a)
        elif isinstance(value, (dict)):
            return TensorUtil._process_dict(value, name, context, dtype=dtype, device=device)
        elif hasattr(value, '__iter__'):
            # Handling iterables which are not lists or tuples => handle them as dict.
            return TensorUtil._process_dict(dict(value), name, context, dtype=dtype, device=device)
        elif hasattr(value, '__dict__'):
            new_val = TensorUtil._process_dict(
                dict(value.__dict__), name, context, dtype=dtype, device=device)
            # Setting all properties manually
            for k, v in new_val.items():
                setattr(value, k, v)
        else:
            raise NoIterationTypeError()

    @staticmethod
    def apply_deep(obj: Any,
                   fnc: Callable[[torch.Tensor], torch.Tensor],
                   memo: Set[Any] = None,
                   path: Optional[str] = None,
                   accepts_path: Optional[bool] = None,
                   ) -> Any:
        """Applies the given function on each tensor found in a object graph.

        Creates a deep copy of each object in the graph while querying it.

        Parameters
        ----------
        obj : Any
            A object containing tensors.
        fnc : Callable[[torch.Tensor], torch.Tensor]
            A function to apply for.
        memo : Set[Any], optional
            Memo of already visitend objects., by default None

        Returns
        -------
        Any
            The altered object.
        """
        if memo is None:
            memo = set()
        if path is None:
            path = ""
        if accepts_path is None:
            # Use inspect to check if the function accepts a path
            accepts_path = False
            import inspect
            sig = inspect.signature(fnc)
            # Check if the function has a path argument or a **kwargs
            if "path" in sig.parameters or "**kwargs" in sig.parameters:
                accepts_path = True

        try:
            if hasattr(obj, "__hash__") and obj.__hash__ is not None:
                if obj in memo:
                    return obj
                else:
                    memo.add(obj)
        except TypeError as err:
            # Unhashable type
            pass
        if isinstance(obj, (str, int, float, complex)):
            return obj
        if isinstance(obj, torch.Tensor):
            args = {}
            if accepts_path:
                args["path"] = path
            ret = fnc(obj, **args)
            return ret
        elif isinstance(obj, list):
            return [TensorUtil.apply_deep(x, fnc=fnc, memo=memo, path=path + f"[{i}]") for i, x in enumerate(obj)]
        elif isinstance(obj, tuple):
            vals = [TensorUtil.apply_deep(
                x, fnc=fnc, memo=memo, path=path + f"[{i}]") for i, x in enumerate(obj)]
            return tuple(vals)
        elif isinstance(obj, set):
            return set([TensorUtil.apply_deep(x, fnc=fnc, memo=memo, path=path + f"[{id(x)}]") for x in obj])
        elif isinstance(obj, OrderedDict):
            return OrderedDict({k: TensorUtil.apply_deep(v, fnc=fnc, memo=memo, path=path + f"[\"{k}\"]") for k, v in obj.items()})
        elif isinstance(obj, dict):
            return {
                k: TensorUtil.apply_deep(v, fnc=fnc, memo=memo, path=path + f"[\"{k}\"]") for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                setattr(obj, k, TensorUtil.apply_deep(
                    v, fnc=fnc, memo=memo, path=path + f".{k}"))
        return obj


SHADOW_TENSOR_WARNING = False
SHADOW_TENSOR_USAGES: FrameSummary = []


def shadow_zeros(x: torch.Tensor) -> torch.Tensor:
    """Debug function which replaces the given Tensor with
    zeros, so the original one will be not part of the compute graph.

    Parameters
    ----------
    x : torch.Tensor
        The original tensor.

    Returns
    -------
    torch.Tensor
        A zero tensor having the same shape and properties.
    """
    global SHADOW_TENSOR_WARNING, SHADOW_TENSOR_USAGES
    tb = _custom_traceback(1)
    summary = extract_stack(tb.tb_frame, 2)
    frame = summary[-1]

    if not SHADOW_TENSOR_WARNING:
        # Warn that shadowing is active.
        filename = relpath(os.getcwd(), frame.filename, is_from_file=False)
        logger.warning("Some tensor(s) are beeing shadowed, check SHADOW_TENSOR_USAGES for all code positions.\n" +
                       f"First position: {filename} Line: {frame.lineno}.")
        SHADOW_TENSOR_WARNING = True
    SHADOW_TENSOR_USAGES.append(frame)

    shape = x.shape
    device = x.device
    dtype = x.dtype
    return torch.zeros(shape, dtype=dtype, device=device)


def shadow_ones(x: torch.Tensor) -> torch.Tensor:
    """Debug function which replaces the given Tensor with
    ones, so the original one will be not part of the compute graph.

    Parameters
    ----------
    x : torch.Tensor
        The original tensor.

    Returns
    -------
    torch.Tensor
        A ones tensor having the same shape and properties.
    """
    global SHADOW_TENSOR_WARNING, SHADOW_TENSOR_USAGES
    tb = _custom_traceback(1)
    summary = extract_stack(tb.tb_frame, 2)
    frame = summary[-1]

    if not SHADOW_TENSOR_WARNING:
        # Warn that shadowing is active.
        filename = relpath(os.getcwd(), frame.filename, is_from_file=False)
        logger.warning("Some tensor(s) are beeing shadowed, check SHADOW_TENSOR_USAGES for all code positions.\n" +
                       f"First position: {filename} Line: {frame.lineno}.")
        SHADOW_TENSOR_WARNING = True
    SHADOW_TENSOR_USAGES.append(frame)

    shape = x.shape
    device = x.device
    dtype = x.dtype
    return torch.ones(shape, dtype=dtype, device=device)


def shadow_identity_2d(x: torch.Tensor) -> torch.Tensor:
    """Debug function which replaces the given Tensor with
    a identity matrix, so the original one will be not part of the compute graph.

    Parameters
    ----------
    x : torch.Tensor
        The original tensor. Shape ([..., B], N, N)

    Returns
    -------
    torch.Tensor
        A identity matrix tensor in 2D
        Shape ([..., B], N, N)
        Where (N, N) will be an identity matrix
    """
    global SHADOW_TENSOR_WARNING, SHADOW_TENSOR_USAGES
    tb = _custom_traceback(1)
    summary = extract_stack(tb.tb_frame, 2)
    frame = summary[-1]

    if not SHADOW_TENSOR_WARNING:
        # Warn that shadowing is active.
        filename = relpath(os.getcwd(), frame.filename, is_from_file=False)
        logger.warning("Some tensor(s) are beeing shadowed, check SHADOW_TENSOR_USAGES for all code positions.\n" +
                       f"First position: {filename} Line: {frame.lineno}.")
        SHADOW_TENSOR_WARNING = True
    SHADOW_TENSOR_USAGES.append(frame)
    if len(x.shape) < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("Need x to be in ([..., B,] N, N) shape.")
    shape = x.shape
    device = x.device
    dtype = x.dtype
    xf, bd = flatten_batch_dims(x, -3)
    B = xf.shape[0]
    eye = torch.eye(x.shape[-1], dtype=dtype,
                    device=device).unsqueeze(0).repeat(B, 1, 1)
    return unflatten_batch_dims(eye, bd)


@torch.jit.script
def flatten_batch_dims(tensor: torch.Tensor, end_dim: int) -> Tuple[torch.Tensor, List[int]]:
    """

    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to flatten.
    end_dim : int
        Maximum batch dimension to flatten (inclusive).

    Returns
    -------
    Tuple[torch.Tensor, Tuple[int]]
        The flattend tensor and the original batch shape.
    """
    ed = end_dim + 1 if end_dim != -1 else None
    batch_shape = tensor.shape[:ed]

    expected_dim = -1 if end_dim >= 0 else abs(end_dim)

    if len(batch_shape) > 0:
        flattened = tensor.flatten(end_dim=end_dim)
    else:
        flattened = tensor.unsqueeze(0)
        if expected_dim > 0:
            missing = expected_dim - len(flattened.shape)
            for _ in range(missing):
                flattened = flattened.unsqueeze(0)
    return flattened, batch_shape


@torch.jit.script
def unflatten_batch_dims(tensor: torch.Tensor, batch_shape: List[int]) -> torch.Tensor:
    """Method to unflatten a tensor, which was previously flattened using flatten_batch_dims.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to unflatten.
    batch_shape : List[int]
        Batch shape to unflatten.

    Returns
    -------
    torch.Tensor
        The unflattened tensor.
    """

    if len(batch_shape) > 0:
        if not isinstance(batch_shape, list):
            batch_shape = list(batch_shape)
        cur_dim = list(tensor.shape[1:])
        new_dims = batch_shape + cur_dim
        return tensor.reshape(new_dims)
    else:
        return tensor.squeeze(0)


def grad_cached(
        device: torch.device = "cpu",
        return_key: str = "return",
        retrieve_key: str = "get_cache",
        clear_key: str = "clear_cache"
) -> bool:
    import inspect
    func_exec_cache = dict()
    if isinstance(device, str):
        device = torch.device(device)

    def format_output(out: Dict[str, List[torch.Tensor]]) -> Dict[str, Union[List[Any], torch.Tensor]]:
        ret = {}
        for k, v in out.items():
            if len(v) == 0:
                continue
            if not isinstance(v[0], torch.Tensor):
                continue
            ret[k] = torch.stack(v, dim=0)
        return ret

    def process_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(device)
        return value

    def decorator(grad_hook: Callable[[torch.Tensor, Any], torch.Tensor]) -> Callable[[torch.Tensor, Any], torch.Tensor]:
        nonlocal func_exec_cache
        if grad_hook is None:
            return None
        params = inspect.signature(grad_hook).parameters

        order_name_mapping = dict()
        for i, (k, v) in enumerate(params.items()):
            order_name_mapping[i] = str(k)
            func_exec_cache[str(k)] = []

        func_exec_cache[return_key] = []

        @wraps(grad_hook)
        def wrapper(*args, **kwargs):
            nonlocal func_exec_cache, order_name_mapping
            if retrieve_key in kwargs and kwargs[retrieve_key]:
                return format_output(func_exec_cache)
            if clear_key in kwargs and kwargs[clear_key]:
                for k in func_exec_cache.keys():
                    func_exec_cache[k] = []
                return None
            for i, a in enumerate(args):
                func_exec_cache[order_name_mapping[i]].append(process_value(a))
            for k, v in kwargs.items():
                func_exec_cache[k].append(process_value(v))

            try:
                fnc_out = grad_hook(*args, **kwargs)
            except Exception as err:
                # Remove already added values
                for i, a in enumerate(args):
                    del func_exec_cache[order_name_mapping[i]][-1]
                for k, v in kwargs.items():
                    del func_exec_cache[k][-1]
                raise err

            func_exec_cache[return_key].append(process_value(fnc_out))
            return fnc_out
        return wrapper
    return decorator


def plot_weight(x: torch.Tensor, title: str = "Weights", cmap: str = "viridis", colorbar: bool = True, **kwargs) -> Any:
    """Plots a 1D or 2D tensor as an image.

    Including the gradients if available.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to plot.

    title : str, optional
        Title of the plot, by default "Weights"

    cmap : str, optional
        Colormap to use, by default "viridis"

    colorbar : bool, optional
        If a colorbar should be displayed, by default True

    Returns
    -------
    Any
        The matplotlib figure.
    """
    org_shape = tuple(x.shape)
    from tools.viz.matplotlib import get_mpl_figure, plot_as_image
    grad = None
    weight = x.detach().clone().cpu().numpy()
    if x.requires_grad or x.grad is not None:
        grad = x.grad.detach().clone().cpu().numpy()
    if len(x.shape) not in [1, 2]:
        raise ValueError("Only 1D or 2D tensors are supported.")
    H, W = None, None
    if len(x.shape) == 1:
        from sympy import divisors
        B = x.shape[0]
        d = torch.tensor(divisors(B)).unsqueeze(1)
        times = torch.tensor(B).unsqueeze(0) / d
        min_d = (times - d).abs().argmin().squeeze()
        if len(min_d.shape) > 0:
            min_d = min_d[0]
        H = d[min_d].squeeze().item()
        W = times[min_d].squeeze().int().item()
        weight = weight.reshape(H, W)
        if grad is not None:
            grad = grad.reshape(H, W)
    else:
        H, W = weight.shape
    target_shape = (H, W)
    rows = 1
    cols = (2 if grad is not None else 1)
    fig, axes = get_mpl_figure(
        rows=rows, cols=cols, ratio_or_img=H / W, ax_mode="1d")
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    plot_as_image(weight, axes=axes[0], colorbar=colorbar,
                  cmap=cmap, variable_name="Weights", **kwargs)
    if grad is not None:
        plot_as_image(grad, axes=axes[1], colorbar=colorbar,
                      cmap=cmap, variable_name="Gradients", **kwargs)

    axw = fig.add_subplot(rows, 1, 1, frameon=False)
    if org_shape != target_shape:
        shape_change = f"[{', '.join([str(s) for s in org_shape])}] -> [{', '.join([str(s) for s in target_shape])}]"
        title = f"{title} {shape_change}"
    axw.set_title(title)
    axw.axis("off")

    return fig


def rowwise_isin(x, y):
    """Checks if the elements of x are in y rowwise.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (N, K)
    y : torch.Tensor
        Tensor of shape (N, L)

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (N, K) where result[n, k] is torch.isin(x[n, k], y[n])
    """

    matches = (x.unsqueeze(2) == y.unsqueeze(1))
    # result: boolean tensor of shape (N, K) where result[n, k] is torch.isin(x[n, k], y[n])
    result = torch.sum(matches, dim=2, dtype=torch.bool)
    return result


def consecutive_indices_string(x: VEC_TYPE, slice_sep: str = "-", item_sep: str = ",") -> str:
    """Formats a 1D tensor of (consecutive) indices into a string representation.

    Indices of similar step size are grouped together. The output is a string
    where each item is a string of the form "StartSlice[-EndSlice-StepSize]".
    StartSlice is the starting index of the group, EndSlice is the ending index, both are inclusive.
    If the group size is smaller than 3, there will be no grouping and the indices will be printed as single items.

    Example:
    x = [0, 1, 2, 3, 5, 7, 9, 11, 15, 16, 19]
    consecutive_indices_string(x) -> "0-3-1,5-11-2,15,16,19"

    Parameters
    ----------
    x : VEC_TYPE
        A 1D tensor of indices.
    slice_sep : str, optional
        Seperator for slices, by default "-"
    item_sep : str, optional
        Seperator for items, by default ","

    Returns
    -------
    str
        String representation of the input tensor.
    """
    from tools.util.format import consecutive_indices_string
    return consecutive_indices_string(x, slice_sep=slice_sep, item_sep=item_sep)


def _is_non_finite(x: torch.Tensor, info: bool = False) -> Union[bool, Tuple[bool, torch.Tensor]]:
    """Checks if a tensor contains non-finite values.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check.
        Shape: ([..., B])
    Returns
    -------
    Union[bool, Tuple[bool, torch.Tensor]]

    bool
        True if the tensor contains non-finite values.


    """
    finite = torch.isfinite(x)
    non_finite = not finite.all()
    if not info:
        return non_finite
    # Get argwhere and return the values
    idx = torch.argwhere(~finite)

    values = x[tuple(idx[:, j] for j in range(len(idx.shape)))]
    if values.shape[0] != 0:
        values = values.unsqueeze(1)
    info = torch.cat([idx, values], dim=-1)
    return non_finite, info


@torch.jit.script
def cummatmul(x: torch.Tensor) -> torch.Tensor:
    """
    Cumulative matrix multiplication along the batch dimension.
    Given matricies {M1, M2, M3, ..., Mn} the result will be {M1, M2 @ M1, M3 @ M2 @ M1, ..., Mn @ ...@ M3 @ M2 @ M1}.
    "@" denotes matrix multiplication.


    Parameters
    ----------
    x : torch.Tensor
        Tensor to cumulatively multiply.
        Further
        Shape: (B, N, N)

    Returns
    -------
    torch.Tensor
        Cumulative matrix multiplication tensor.
        Shape: (B, N, N)
    """
    if len(x.shape) != 3:
        raise ValueError("Input tensor must have shape (B, N, N)")
    B, N, M = x.shape
    if N != M:
        raise ValueError("Input matrix must be square. Shape: (B, N, N)")
    cum = x.clone()
    for i in range(1, B):
        cum[i] = torch.matmul(x[i], cum[i - 1])
    return cum


def is_non_finite(x: torch.Tensor, info: bool = False) -> Union[bool, Tuple[bool, Dict[str, torch.Tensor]]]:
    """Checks if a tensor contains non-finite values.
    Will also check the gradients if available.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check.
        Shape: ([..., B])
    Returns
    -------
    Union[bool, Tuple[bool, torch.Tensor]]

    bool
        True if the tensor contains non-finite values.
        E.g. if any value is nan or inf.

    Tuple[bool, Dict[str, torch.Tensor]]
        If info is True, a dictionary with the keys "data" and "grad" will be returned.
        The values are the non-finite values of the data and gradients.
        Shape: ([N, D+1])
        Where N is the number of non-finite values and D is the number of dimensions of the tensor.
        Last column is the value itself.
    """
    non_finite = _is_non_finite(x, info=info)
    value_info = None
    if info:
        value_info = non_finite[1]
        non_finite = non_finite[0]
    grad_non_finite = False
    grad_info = None
    if x.grad is not None:
        grad_non_finite = _is_non_finite(x.grad, info=info)
        if info:
            grad_info = grad_non_finite[1]
            grad_non_finite = grad_non_finite[0]
    if not info:
        return non_finite or grad_non_finite
    combined_info = dict(
        data=value_info,
        grad=grad_info
    )
    return non_finite or grad_non_finite, combined_info


def buffered(gen: Generator,
             gather_fnc: Callable[[Any], torch.Tensor],
             execute_fnc: Callable[[torch.Tensor], torch.Tensor],
             assemble_fnc: Callable[[Any, torch.Tensor], Any],
             buffer_size: int = -1) -> Generator[Any, None, None]:
    """Bufferes values of a generator and applies a function to the buffered values.
    Will itself return a generator that yields the assembled output of the execution.

    Applicable to perform costly operations on a batch of data.

    Parameters
    ----------
    gen : Generator
        The generator function to which the execution will be applied.

    gather_fnc : Callable[[Any], Union[torch.Tensor, Generator[torch.Tensor, None, None]]]
        The function that extracts the data from the generator output and converts it to a tensor.

        Parameters
        ----------
        data : Any
            The data output of the generator.

        Returns
        -------
        Union[torch.Tensor, Generator[torch.Tensor, None, None]]
            The gathered tensor of shape (BI, ...).
            Or a generator that yields tensors of shape (BI, ...).

    execute_fnc : Callable[[torch.Tensor], torch.Tensor]
        The function that will be applied to the gathered batched data.

        Parameters
        -------

        tensor : torch.Tensor
            The gathered tensor of shape (B, ...).
            Where B is the batch size / a "cated" version of BI up to the batch size length.. B might be stacked out of multiple BI elements from the generator.

        Returns
        -------
        torch.Tensor
            The executed tensor of shape (B, ...).


        Should accept a tensor of shape (B, ...) and return a tensor of shape (B, ...).


    assemble_fnc : Callable[[Any, torch.Tensor, int], Any]
        The function that assembles the output of the execution back to the desired output format.

        Parameters
        ----------
        data : Any
            The original data output of the generator.

        tensor : torch.Tensor
            The executed tensor of shape (BI, ...).
            Where BI is the data accociated with the original data output, transformed by gather_fnc and execute_fnc.

        index : int
            The index of the item in the original data if execute_fnc was a generator itself. Otherwise 0.

        Returns
        -------
        Any
            The assembled output of the execution.
            Each will be yielded by the generator.

    buffer_size : int, optional
        The size of the buffer, by default -1.
        -1 means a buffer of infinite size, the generator is fully consumed before the execution of execute_fnc.

    Yields
    -------
    Any
        The assembled output of the generator.
    """

    # Retrieve the first batch of data
    _current_data = []
    _current_buffer = []
    _current_data_gen_index = []
    _current_item_shape = []
    _current_buffer_size = 0

    def process_items(
            _data,
            _data_gen_index,
            _buffer,
            _item_shape,
    ):
        b = torch.cat(_buffer, dim=0)
        num_batches = 1
        if buffer_size > 0:
            num_batches = math.ceil(b.shape[0] / buffer_size)
        res = []
        for i in range(num_batches):
            start = i * buffer_size
            end = min(start + buffer_size, b.shape[0])
            exec_b = execute_fnc(b[start:end])
            res.append(exec_b)
        res = torch.cat(res, dim=0)
        start = 0
        for i, (org_data, idx, shp) in enumerate(zip(_data, _data_gen_index, _item_shape)):
            item_start = start
            item_end = start + shp[0]
            start = item_end
            yield assemble_fnc(org_data, res[item_start:item_end], idx)

    for i, data in enumerate(gen):

        extracted_data = gather_fnc(data)
        if isinstance(extracted_data, torch.Tensor):
            _current_data.append(data)
            _current_buffer.append(extracted_data)
            _current_data_gen_index.append(0)
            shp = extracted_data.shape
            _current_item_shape.append(shp)
            _current_buffer_size += shp[0]
        elif isinstance(extracted_data, Generator):
            for j, d in enumerate(extracted_data):
                _current_data.append(data)
                _current_data_gen_index.append(j)
                _current_buffer.append(d)
                shp = d.shape
                _current_item_shape.append(shp)
                _current_buffer_size += shp[0]

        if buffer_size > 0 and _current_buffer_size >= buffer_size:
            for item in process_items(_current_data,
                                      _current_data_gen_index,
                                      _current_buffer, _current_item_shape):
                yield item
            # Reset the buffer
            _current_data = []
            _current_data_gen_index = []
            _current_buffer = []
            _current_item_shape = []
            _current_buffer_size = 0

    # Process the remaining data
    if len(_current_data) > 0:
        for item in process_items(_current_data, _current_data_gen_index, _current_buffer, _current_item_shape):
            yield item


def parse_dtype(_dtype_or_str: Union[str, torch.dtype, np.dtype]) -> torch.dtype:
    """Parses a string or torch.dtype to a torch.dtype.

    Parameters
    ----------
    _dtype_or_str : Union[str, torch.dtype]
        The dtype or string to parse.

    Returns
    -------
    torch.dtype
        The parsed dtype.
    """
    if isinstance(_dtype_or_str, np.dtype):
        return numpy_to_torch_dtype(_dtype_or_str)
    if isinstance(_dtype_or_str, str):
        if _dtype_or_str.startswith("torch."):
            _dtype_or_str = _dtype_or_str.replace("torch.", "")
        dt = getattr(torch, _dtype_or_str)
        if dt is None:
            raise ValueError(f"Invalid dtype string: {_dtype_or_str}")
        if not isinstance(dt, torch.dtype):
            raise ValueError(
                f"Invalid dtype string: {_dtype_or_str}, must be a valid torch dtype.")
        return dt
    elif isinstance(_dtype_or_str, torch.dtype):
        return _dtype_or_str
    else:
        raise ValueError(f"Invalid dtype: {_dtype_or_str}")


def sort_module_list(module_list: ModuleList, order: List[torch.nn.Module]) -> ModuleList:
    """Sorts a ModuleList according to the given order.

    Performs the order in-place, so the original ModuleList will be modified.

    May raise a ValueError if the order is not valid, eg. some elements are missing.

    Parameters
    ----------
    module_list : ModuleList
        The ModuleList to sort.
    order : List[torch.nn.Module]
        The order to sort the ModuleList by.

    Returns
    -------
    ModuleList
        The sorted ModuleList.
    """
    current_order = [order.index(m) for m in module_list]
    sorted_order = sorted(current_order)
    if current_order == sorted_order:
        return module_list
    needed_order_d = {order.index(m): m for m in module_list}
    for i in range(len(order)):
        item = needed_order_d.get(i)
        current_index = list(module_list).index(item)
        v = module_list.pop(current_index)
        module_list.insert(i, v)
    return module_list


def on_load_checkpoint(
    module: torch.nn.Module,
    state_dict: Dict[str, Any],
    allow_loading_unmatching_parameter_sizes: bool = False,
    load_missing_parameters_as_defaults: bool = False,
):
    if allow_loading_unmatching_parameter_sizes or load_missing_parameters_as_defaults:
        current_state_dict = module.state_dict(keep_vars=True)
        loaded_state_dict = state_dict
        loaded_keys = set(loaded_state_dict.keys())
        current_keys = set(current_state_dict.keys())
        missing_keys = loaded_keys - current_keys

        shape_changes = []
        missing_parameters = []
        for k in current_state_dict:
            if k in loaded_state_dict:
                current_shape = current_state_dict[k].shape
                loaded_shape = loaded_state_dict[k].shape
                if allow_loading_unmatching_parameter_sizes:
                    if current_shape != loaded_shape:
                        # Set the current parameter with an empty tensor like the loaded one
                        p = current_state_dict[k]
                        p.data = torch.empty_like(
                            loaded_state_dict[k], device=p.device, dtype=p.dtype)
                        shape_changes.append(
                            "{}: {} -> {}".format(k, list(current_shape), list(loaded_shape)))
            else:
                if load_missing_parameters_as_defaults:
                    loaded_state_dict[k] = current_state_dict[k].detach(
                    ).clone()
                    missing_parameters.append("{}: {}".format(
                        k, list(current_state_dict[k].shape)))
        if allow_loading_unmatching_parameter_sizes:
            if len(shape_changes) > 0:
                logger.info("Loaded model has different parameter sizes than current model. The following parameters have been changed: \n{}".format(
                    ",\n".join(shape_changes)))
        if load_missing_parameters_as_defaults:
            if len(missing_parameters) > 0:
                logger.info("Loaded model is missing parameters. The following parameters have been added: \n{}".format(
                    ",\n".join(missing_parameters)))

        if len(missing_keys) > 0:
            logger.warning(
                "Loaded model is missing keys: {}".format(missing_keys))
