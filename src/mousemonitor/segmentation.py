from typing import Tuple

import numpy as np
from numba import njit


@njit
def get_full_transition_mat(n_states: int, penalty: float) -> np.ndarray:
    # transition_penalty_mat, shape (n_states, n_states)
    transition_penalty_mat = np.full((n_states, n_states), penalty, dtype=np.float64)
    for i in range(n_states):
        transition_penalty_mat[i, i] = 0.0
    return transition_penalty_mat


@njit
def min_plus_matvec(M: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # out, shape (n_states,)
    # arg_out, shape (n_states,)
    n = M.shape[0]
    out = np.zeros((n,), dtype=np.float64)
    arg_out = np.zeros((n,), dtype=np.int64)
    for k_row in range(n):
        arg_minimum = np.argmin(M[k_row] + v)
        out[k_row] = M[k_row, arg_minimum] + v[arg_minimum]
        arg_out[k_row] = arg_minimum
    return out, arg_out


@njit
def get_state_sequence(costs: np.ndarray, transition_mat: np.ndarray) -> np.ndarray:
    # costs, shape (n_samples, n_states)
    # transition_mat, shape (n_states, n_states)
    # optimal_state_sequence, shape (n_samples,)
    n_samples, n_states = costs.shape
    soc_array = np.empty((n_samples + 1, n_states), dtype=np.float64)
    state_array = np.empty((n_samples + 1, n_states), dtype=np.int64)
    soc_array[0] = 0
    state_array[0] = -1

    # Forward loop
    for end in range(1, n_samples + 1):
        soc_vec, state_vec = min_plus_matvec(M=transition_mat, v=soc_array[end - 1])
        soc_array[end] = soc_vec + costs[end - 1]
        state_array[end] = state_vec

    # Backtracking
    end = n_samples
    state = np.argmin(soc_array[end])
    optimal_state_sequence = np.empty(n_samples, dtype=np.int64)
    while end > 0:
        optimal_state_sequence[end - 1] = state
        state = state_array[end, state]
        end -= 1
    return optimal_state_sequence


def opt_state_sequence_binary(
    signal: np.ndarray, penalty: float
) -> Tuple[np.ndarray, np.ndarray]:
    # signal, shape (n_samples,)
    # penalty, > 0
    # bkps, shape (n_bkps+1,)
    # opt_state_sequence, shape (n_samples,)
    costs = np.abs(np.c_[signal - 1, signal + 1])
    transition_mat = get_full_transition_mat(n_states=2, penalty=penalty)
    opt_state_sequence = get_state_sequence(costs, transition_mat)
    bkps = np.nonzero(np.diff(opt_state_sequence))[0] + 1
    return bkps, opt_state_sequence
