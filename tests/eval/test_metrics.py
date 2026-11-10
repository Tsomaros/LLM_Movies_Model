import math

import numpy as np
import pytest

from llm_movies.eval.metrics import revenue_metrics, score_metrics

# Pinball per row = mean over q in (0.1, 0.5, 0.9) of max(q*d, (q-1)*d), d = y - pred.

# Revenue, 4 rows (true log, [q10, q50, q90]):
#   r1 y=10 [10, 10.5, 12]    |q50-y|=0.5 within, covered (true == q10 boundary), width e^2
#   r2 y=12 [10, 13, 16]      |q50-y|=1   not within (>ln2), covered, width e^6
#   r3 y=14 [13.5, 14, 14]    |q50-y|=0   within, covered (true == q90 boundary), width e^0.5
#   r4 y=16 [10, 12, 15]      |q50-y|=4   not within, not covered (16>15), width e^5
REV_PRED = np.array([[10, 10.5, 12], [10, 13, 16], [13.5, 14, 14], [10, 12, 15]], dtype=float)
REV_TRUE = np.array([10, 12, 14, 16], dtype=float)


def test_revenue_metrics_hand_worked():
    m = revenue_metrics(REV_PRED, REV_TRUE)
    # mae_log = (0.5 + 1 + 0 + 4) / 4 = 1.375
    assert m["mae_log"] == pytest.approx(1.375)
    # within_2x = 2 of 4 (0.5 and 0 are <= 0.693; 1 and 4 are not)
    assert m["within_2x"] == pytest.approx(0.5)
    # coverage_80 = 3 of 4 (r4 misses); r1 and r3 sit exactly on q10 / q90 and count as covered
    assert m["coverage_80"] == pytest.approx(0.75)
    # widths e^2, e^6, e^0.5, e^5; sorted e^.5,e^2,e^5,e^6; median = (e^2 + e^5) / 2
    # (the mean, (e^.5+e^2+e^5+e^6)/4, is about 143, the median about 78)
    assert m["median_width_x"] == pytest.approx((math.exp(2) + math.exp(5)) / 2)
    # r1 d=[0,-.5,-2]: 0+.25+.2=.45; r2 d=[2,-1,-4]: .2+.5+.4=1.1; r3 d=[.5,0,0]: .05+0+0=.05;
    # r4 d=[6,4,1]: .6+2+.9=3.5; pinball = (.45+1.1+.05+3.5) / (4*3) = 5.1 / 12 = 0.425
    assert m["pinball"] == pytest.approx(0.425)


# Score, 4 rows (true, [q10, q50, q90], weight):
#   r1 y=6   [4, 6.5, 8]   w=1  err .5, covered, width 4
#   r2 y=7   [7, 7, 8]     w=1  err 0,  covered (true == q10), width 1
#   r3 y=8   [6, 7, 7.5]   w=4  err 1,  not covered (8>7.5), width 1.5
#   r4 y=6.5 [4, 6, 6.5]   w=2  err .5, covered (true == q90), width 2.5
SC_PRED = np.array([[4, 6.5, 8], [7, 7, 8], [6, 7, 7.5], [4, 6, 6.5]], dtype=float)
SC_TRUE = np.array([6, 7, 8, 6.5], dtype=float)
SC_W = np.array([1, 1, 4, 2], dtype=float)


def test_score_metrics_hand_worked():
    m = score_metrics(SC_PRED, SC_TRUE, SC_W)
    # weighted mae = (1*.5 + 1*0 + 4*1 + 2*.5) / 8 = 5.5 / 8 = 0.6875
    assert m["mae"] == pytest.approx(0.6875)
    # coverage is unweighted: 3 of 4 (a weighted version would give (1+1+0+2)/8 = 0.5)
    assert m["coverage_80"] == pytest.approx(0.75)
    # widths 4, 1, 1.5, 2.5; sorted 1, 1.5, 2.5, 4; median = 2.0 (the mean would be 2.25)
    assert m["median_width"] == pytest.approx(2.0)
    # row sums: r1 d=[2,-.5,-2] .2+.25+.2=.65; r2 d=[0,0,-1] 0+0+.1=.1;
    # r3 d=[2,1,.5] .2+.5+.45=1.15; r4 d=[2.5,.5,0] .25+.25+0=.5
    # weighted pinball = (1*.65 + 1*.1 + 4*1.15 + 2*.5) / (3 * 8) = 6.35 / 24
    assert m["pinball"] == pytest.approx(6.35 / 24)


def test_exactly_2x_counts_within():
    # true 0 keeps |q50 - true| exactly ln(2) in floating point; both directions count
    true = np.zeros(4)
    pred = np.array(
        [
            [-1, math.log(2), 1],  # exactly 2x over
            [-1, -math.log(2), 1],  # exactly 2x under
            [-1, math.log(2) + 1e-6, 1],  # just over 2x
            [-1, -math.log(2) - 1e-6, 1],  # just under 2x
        ]
    )
    assert revenue_metrics(pred, true)["within_2x"] == pytest.approx(0.5)


def test_weights_change_score_mae():
    unweighted = score_metrics(SC_PRED, SC_TRUE, np.ones(4))["mae"]
    weighted = score_metrics(SC_PRED, SC_TRUE, SC_W)["mae"]
    assert unweighted == pytest.approx(0.5)  # (.5+0+1+.5)/4
    assert weighted == pytest.approx(0.6875)
    assert weighted != pytest.approx(unweighted)


def test_weights_change_score_pinball():
    # equal weights: (.65+.1+1.15+.5) / (4*3) = 2.4/12 = 0.2, unlike the weighted 6.35/24 = 0.2646
    unweighted = score_metrics(SC_PRED, SC_TRUE, np.ones(4))["pinball"]
    assert unweighted == pytest.approx(0.2)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        revenue_metrics(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError):
        score_metrics(np.empty((0, 3)), np.empty(0), np.empty(0))
