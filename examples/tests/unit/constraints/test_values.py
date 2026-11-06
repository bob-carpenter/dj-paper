"""Test hand-computed values: forward, inverse and log-det of each constraint at known points."""
import jax.numpy as jnp
import numpy as np
import pytest

from util import box, positive, simplex, zero_sum

POSITIVE = positive(shape=(3,))
BOX = box(lower=-1.0, upper=3.0, shape=(3,))
BOX_INF = box(lower=jnp.array([-1.0, 2.0, -jnp.inf, -jnp.inf]),
              upper=jnp.array([3.0, jnp.inf, 1.0, jnp.inf]), shape=(4,))
ZERO_SUM = zero_sum(4)
SIMPLEX = simplex(4)
L2, L3 = np.log(2.0), np.log(3.0)

KNOWN = [  #       c          x                     y                       log-det
    pytest.param(POSITIVE, [0.0, L2, L3],         [1.0, 2.0, 3.0],         np.log(6.0),
                 id="positive:x=log(1,2,3)->y=(1,2,3)"),
    pytest.param(POSITIVE, [L2, L2, L2],          [2.0, 2.0, 2.0],         np.log(8.0),
                 id="positive:x=log(2,2,2)->y=(2,2,2)"),
    pytest.param(BOX,      [0.0, 0.0, 0.0],       [1.0, 1.0, 1.0],         0.0,
                 id="box:x=(0,0,0)->y=(1,1,1)"),
    pytest.param(BOX,      [-L3, 0.0, L3],        [0.0, 1.0, 2.0],         np.log(9/16),
                 id="box:x=(-log3,0,log3)->y=(0,1,2)"),
    pytest.param(BOX_INF,  [0.0, 0.0, 0.0, 0.0],  [1.0, 3.0, 0.0, 0.0],    0.0,
                 id="box_inf:x=(0,0,0,0)->y=(1,3,0,0)"),
    pytest.param(BOX_INF,  [L3, L2, -L2, 5.0],    [2.0, 4.0, -1.0, 5.0],   L3,
                 id="box_inf:x=(log3,log2,-log2,5)->y=(2,4,-1,5)"),
    pytest.param(ZERO_SUM, [1.0, 1.0, 1.0],       [1/2, 1/2, 1/2, -3/2],   0.0,
                 id="zero_sum:x=(1,1,1)->y=(1,1,1,-3)/2"),
    pytest.param(ZERO_SUM, [1.0, 0.0, 0.0],       [5/6, -1/6, -1/6, -1/2], 0.0,
                 id="zero_sum:x=(1,0,0)->y=(5,-1,-1,-3)/6"),
    pytest.param(SIMPLEX,  [0.0, 0.0, 0.0],       [1/4, 1/4, 1/4, 1/4],    np.log(1/128),
                 id="simplex:x=(0,0,0)->y=(1,1,1,1)/4"),
    pytest.param(SIMPLEX,  [3*L2/2, L2/2, -L2/2], [1/2, 1/4, 1/8, 1/8],    np.log(1/256),
                 id="simplex:x=log(2)(3,1,-1)/2->y=(4,2,1,1)/8"),
]


@pytest.mark.parametrize("c, x, y, log_det", KNOWN)
def test_forward_at_x__gives_hand_computed_y(c, x, y, log_det):
    np.testing.assert_allclose(c(jnp.array(x)), y, atol=1e-10)


@pytest.mark.parametrize("c, x, y, log_det", KNOWN)
def test_inverse_at_y__gives_hand_computed_x(c, x, y, log_det):
    np.testing.assert_allclose(c.inverse(jnp.array(y)), x, atol=1e-10)


@pytest.mark.parametrize("c, x, y, log_det", KNOWN)
def test_log_det_at_x__gives_hand_computed_value(c, x, y, log_det):
    np.testing.assert_allclose(c.jacobian(jnp.array(x)), log_det, atol=1e-10)
