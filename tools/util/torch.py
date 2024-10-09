import decimal
from functools import wraps
import logging
import os
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
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from typing import Callable, Tuple
from tools.util.reflection import class_name
from tools.util.format import _custom_traceback
from tools.logger.logging import tools_logger as logger
from traceback import FrameSummary, extract_stack


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


def tensorify(input: NUMERICAL_TYPE,
              dtype: Optional[torch.dtype] = None,
              device: Optional[torch.device] = None,
              requires_grad: bool = False) -> torch.Tensor:
    """
    Assuring that input is a tensor by converting it to one.
    Accepts tensors or ndarrays.

    Parameters
    ----------
    input : Union[torch.Tensor, np.generic, int, float, complex, Decimal]
        The input

    dtype : Optional[torch.dtype]
        Dtype where input should be belong to. If it differs it will cast the type.
        By default its None and the dtype wont be changed.

    device : Optional[torch.device]
        Device where input should be on / send to. If it differs it will move.
        By default its None and the device wont be changed.

    requires_grad : bool
        If the created tensor requires gradient, Will be only considered if input is not already a tensor!. Defaults to false.

    Returns
    -------
    torch.Tensor
        The created tensor.
    """
    if isinstance(input, torch.Tensor):
        if (dtype and input.dtype != dtype) or (device and input.device != device):
            input = input.to(dtype=dtype, device=device)
        return input
    return torch.tensor(input, dtype=dtype, device=device, requires_grad=requires_grad)


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
                return v
            args = [process(arg) for arg in args]
            kwargs = {k: process(v) for k, v in kwargs.items()}
            ret = fnc(*args, **kwargs)
            if keep_output or not is_numpy:
                return ret
            return numpyify(ret)
        return wrapper
    return decorator


def tensorify_image(image: VEC_TYPE,
                    dtype: Optional[torch.dtype] = None,
                    device: Optional[torch.device] = None,
                    requires_grad: bool = False
                    ) -> torch.Tensor:
    """Converts an image to a torch tensor.
    If its already a tensor, it will be returned as is, possibly with changed dtype, device or requires_grad.

    If the image is a numpy array, it will be converted to a tensor with the shape ([B], C, H, W) or (C, H, W) depending on the shape of the input.
    Assumes that the image is in the shape (H, W, C) or (B, H, W, C) for numpy arrays.

    Parameters
    ----------
    image : VEC_TYPE
        Image to convert to a tensor.
    dtype : Optional[torch.dtype], optional
        Dtype for the tensor, by default None
    device : Optional[torch.device], optional
        Device for the tensor, by default None
    requires_grad : bool, optional
        If tensor should require gradients, by default False

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    image = tensorify(image, dtype=dtype, device=device,
                      requires_grad=requires_grad)
    if is_tensor:
        return image
    # Change the shape to ([B], C, H, W)
    if len(image.shape) == 4:
        return image.permute(0, 3, 1, 2)
    # Change the shape to (C, H, W)
    if len(image.shape) == 3:
        return image.permute(2, 0, 1)
    return image


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
        np.bool: torch.bool,
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
        torch.bool: np.bool,
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


def batched_exec(*input,
                 func: Callable[[torch.Tensor], torch.Tensor],
                 batch_size: int,
                 progress_bar: bool = False,
                 pf: Optional[ProgressFactory] = None,
                 **kwargs
                 ) -> torch.Tensor:
    """Execute a function in batches.

    Parameters
    ----------
    input : List[torch.Tensor]
        Should be a list of tensors which have a common batch dimension.
        (B, ...) this is the first dimension and will be used to batch the execution.

    func : Callable[[torch.Tensor], torch.Tensor]
        Function to execute. Should take a tensors as input and return a tensor as output.

    batch_size : int
        The batch size to use for the execution.

    progress_bar : bool, optional
        If a progress bar should be displayed, by default False

    pf : Optional[ProgressFactory], optional
        Progress factory to reuse the progress bar, by default None

    Returns
    -------
    torch.Tensor
        Concatenated results of the function execution.
    """
    results = []
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
            slices), desc=f"Batched Execution: {class_name(func)}", tag=f"BATCH_EXEC_" + class_name(func), is_reusable=True)

    for s in slices:
        _ins = [i[s] for i in input]
        r = func(*_ins, **kwargs)
        results.append(r)
        if progress_bar:
            bar.update(1)
    return torch.cat(results, dim=0)


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
    E = values.shape
    S = search.shape
    res = values[..., None].repeat(*tuple(1 for _ in range(len(E))), *
                                   S) == search[None, ...].repeat(*E, *tuple((1 for _ in range(len(S)))))
    out = torch.zeros(tuple(S), dtype=torch.int)
    out.fill_(-1)
    aw = torch.argwhere(res)
    search_found, where_inverse = torch.unique(aw[:, -1], return_inverse=True)
    for i, s in enumerate(search_found):
        widx = aw[where_inverse == i][0]  # Select first match
        out[s] = widx[0]
    return out


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
    fig, axes = get_mpl_figure(rows=rows, cols=cols, ratio_or_img=H / W, ax_mode="1d")
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    plot_as_image(weight, axes=axes[0], colorbar=colorbar, cmap=cmap, variable_name="Weights", **kwargs)
    if grad is not None:
        plot_as_image(grad, axes=axes[1], colorbar=colorbar, cmap=cmap, variable_name="Gradients", **kwargs)
    
    axw = fig.add_subplot(rows, 1, 1, frameon=False)
    if org_shape != target_shape:
        shape_change = f"[{', '.join([str(s) for s in org_shape])}] -> [{', '.join([str(s) for s in target_shape])}]"
        title = f"{title} {shape_change}"
    axw.set_title(title)
    axw.axis("off")

    return fig