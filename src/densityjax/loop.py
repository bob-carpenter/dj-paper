import jax
import jax.numpy as jnp
import jax.scipy.signal


def recurrence(f, init, *xs, length=None, reverse=False, has_aux=False):
    """
    Run a p-th order recurrence with jax.lax.scan.

    Starting from p known values, each new value is computed from the
    previous p values plus optional per-step inputs::

        x[t] = f(lags, *inputs[t])   where lags = [x[t-1], x[t-2], ..., x[t-p]]

    Parameters
    ----------
    f : function
        Called as ``f(lags, *inputs_t)`` and should return the new value.
        ``lags`` has shape (p, ...) and is ordered nearest first:
        ``lags[0]`` is x[t-1], so ``phi @ lags`` is an AR(p) mean.
    init : array or pytree
        The p starting values, in time order, shape (p, ...). A pytree
        (e.g. a dict) updates several series together; ``f`` then receives
        and returns that same structure.
    *xs : arrays, optional
        Input arrays, each of shape (n, ...). Step i passes
        ``xs[0][i], xs[1][i], ...`` to ``f`` as extra arguments.
    length : int, optional
        Number of steps n. Only needed when no ``xs`` are given.
    reverse : bool, default False
        Run backward in time. ``init`` is then the last p values and
        ``lags[0]`` is x[t+1]. The result is still in time order.
    has_aux : bool, default False
        If True, ``f`` returns ``(new_value, aux)``, and the aux values are
        stacked and returned as well.

    Returns
    -------
    series : array or pytree
        The full series in time order, ``init`` included, shape (p + n, ...).
    aux : array or pytree
        Only returned if ``has_aux`` is True. The stacked aux values,
        shape (n, ...).

    Examples
    --------
    Fibonacci numbers:

    >>> recurrence(lambda lags: lags[0] + lags[1], jnp.array([0, 1]), length=6).tolist()
    [0, 1, 1, 2, 3, 5, 8, 13]
    """
    tmap = jax.tree.map
    window = init if reverse else tmap(lambda a: jnp.asarray(a)[::-1], init)

    def step(window, x_t):
        out = f(window, *x_t)
        new, aux = out if has_aux else (out, None)
        window = tmap(lambda w, n: jnp.concatenate([jnp.asarray(n)[None], w[:-1]]), window, new)
        return window, (new, aux)

    _, (new, aux) = jax.lax.scan(step, window, xs, length=length, reverse=reverse)
    series = tmap(lambda i, n: jnp.concatenate([n, i] if reverse else [i, n]), init, new)
    return (series, aux) if has_aux else series


def lagged_sum(kernel, x, first_lag=0, method="direct"):
    """
    Weighted sum of earlier input values (a causal convolution).

    Each output is a fixed weighted sum of past values of x. Nothing is fed
    back, so all time points are computed at once::

        y[t] = kernel[0] * x[t - first_lag] + kernel[1] * x[t - first_lag - 1] + ...

    Values of x before its start count as 0.

    Parameters
    ----------
    kernel : array
        The weights, shape (..., d). ``kernel[0]`` is the weight for lag
        ``first_lag``, ``kernel[1]`` for lag ``first_lag + 1``, and so on.
    x : array
        The input series, shape (..., T). Time is the last axis (in
        ``recurrence`` it is the first).
    first_lag : int, default 0
        How many of the most recent steps to skip. Must be a non-negative Python 
        int 0 includes x[t] itself; use 1 to predict x[t] from its past only.
    method : {"direct", "fft"}, default "direct"
        Use "fft" for kernels with hundreds of entries: much faster, same
        result up to rounding.

    Returns
    -------
    array
        y, with the same length T as x along the last axis. Leading axes of
        ``kernel`` and ``x`` broadcast: kernel (d,) with x (G, T) applies one
        kernel to G series; kernel (G, d) with x (T,) applies G kernels to
        one series; kernel (G, d) with x (G, T) applies kernel g to series g.

    Notes
    -----
    The first ``first_lag + d - 1`` values are partial sums that include the
    zeros before the start of x. Drop them when fitting if that matters.

    Examples
    --------
    Each input leaves a half-size echo one step later:

    >>> x = jnp.array([1., 0, 0, 0, 2, 0])
    >>> lagged_sum(jnp.array([1., 0.5]), x).tolist()
    [1.0, 0.5, 0.0, 0.0, 2.0, 1.0]

    With ``first_lag=1``, the same echo starts one step later:

    >>> lagged_sum(jnp.array([1., 0.5]), x, first_lag=1).tolist()
    [0.0, 1.0, 0.5, 0.0, 0.0, 2.0]
    """
    if first_lag < 0:
        raise ValueError(f"first_lag must be >= 0, got {first_lag}")
    
    T = x.shape[-1]

    def one(k, row):
        k = jnp.concatenate([jnp.zeros(first_lag, k.dtype), k])
        full = jax.scipy.signal.convolve(row, k, mode="full", method=method,
                                         precision=jax.lax.Precision.HIGHEST)
        return full[:T]

    return jnp.vectorize(one, signature="(d),(t)->(t)")(kernel, x)
