import logging
import numpy as np
import torch

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_float_torch_tensors(tensor, expect_tensor, relative_tol=1e-4, absolute_tol=1e-5, error_tol=1e-4):
    """
    Compare if two PyTorch Tensors are approximately equal in value.

    Functions:
        1. If two tensors are completely identical, return True directly.
        2. Otherwise, use torch.isclose to determine if they are within the given relative/absolute error ranges.
        3. If all elements are within the error ranges, return True.
        4. If some elements do not meet the criteria, calculate the error ratio. Return True if below error_tol
        threshold, otherwise False.

    Parameters:
        tensor (torch.Tensor): Actual result tensor.
        expect_tensor (torch.Tensor): Expected result tensor.
        relative_tol (float): Relative error tolerance (default 1e-4).
        absolute_tol (float): Absolute error tolerance (default 1e-5).
        error_tol (float): Threshold for the proportion of elements exceeding error tolerance (default 1e-4).

    Returns:
        bool: True if tensors meet error conditions, False otherwise.
    """
    if torch.equal(tensor, expect_tensor):
        return True

    compare_tensor = torch.isclose(tensor, expect_tensor, atol=absolute_tol, rtol=relative_tol)
    if torch.all(compare_tensor):
        return True

    correct_count = float(compare_tensor.sum())
    ele_count = compare_tensor.numel()
    err_ratio = (ele_count - correct_count) / ele_count
    return err_ratio < error_tol


def compare_float_np_arrays(arr1, arr2, relative_tol=1e-4, absolute_tol=1e-5, error_tol=1e-4):
    """
    Compare if two NumPy arrays are approximately equal in value, and print the first 100 error elements when
    differences are too large.

    Functions:
        1. Convert input arrays to float32.
        2. Use numpy.isclose to determine if they are within the given relative/absolute error ranges.
        3. If all elements are within the error ranges, return True.
        4. If some elements do not meet the criteria, calculate the error ratio. Return True if below error_tol
        threshold, otherwise False.
        5. When the error ratio exceeds the threshold, print the first 100 differing elements' indices, expected values,
        actual values, and absolute errors.

    Parameters:
        arr1 (np.ndarray): Actual result array.
        arr2 (np.ndarray): Expected result array.
        relative_tol (float): Relative error tolerance (default 1e-4).
        absolute_tol (float): Absolute error tolerance (default 1e-5).
        error_tol (float): Threshold for the proportion of elements exceeding error tolerance (default 1e-4).

    Returns:
        bool: True if arrays meet error conditions, False otherwise.
    """
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    compare_mask = np.isclose(arr1, arr2, atol=absolute_tol, rtol=relative_tol)
    if np.all(compare_mask):
        return True

    total_elements = arr1.size
    correct_count = np.sum(compare_mask)
    error_ratio = (total_elements - correct_count) / total_elements

    if error_ratio >= error_tol:
        diff_indices = np.argwhere(compare_mask == False)
        for idx in diff_indices[:100]:
            idx_tuple = tuple(idx)
            logger.error(f"index {idx_tuple} expected={arr1[idx_tuple]:.9f}, "
                       f"actual={arr2[idx_tuple]:.9f}, "
                       f"adiff={abs(arr1[idx_tuple] - arr2[idx_tuple]):.6f}")
    return error_ratio < error_tol