import math

import numpy as np
import pytest

from llm_movies.eval.metrics import revenue_metrics, score_metrics

# Pinball per row = mean over q in (0.1, 0.5, 0.9) of max(q*d, (q-1)*d), d = y - pred.

# Revenue, 4 rows (true log, [q10, q50, q90]):
#   r1 y=10 [9, 10.5, 12]    |q50-y|=0.5 within, covered, width e^3
#   r2 y=12 [10, 13, 16]     |q50-y|=1   not within (>ln2), covered, width e^6
#   r3 y=14 [13.5, 14, 14.5] |q50-y|=0   within, covered, width e^1
#   r4 y=16 [10, 12, 15]     |q50-y|=4   not within, not covered (16>15), width e^5
REV_PRED = np.array([[9, 10.5, 12], [10, 13, 16], [13.5, 14, 14.5], [10, 12, 15]], dtype=float)
REV_TRUE = np.array([10, 12, 14, 16], dtype=float)


def test_revenue_metrics_hand_worked():
    m = revenue_metrics(REV_PRED, REV_TRUE)
    # mae_log = (0.5 + 1 + 0 + 4) / 4 = 1.375
    assert m["mae_log"] == pytest.approx(1.375)
    # within_2x = 2 of 4 (0.5 and 0 are <= 0.693; 1 and 4 are not)
    assert m["within_2x"] == pytest.approx(0.5)
    # coverage_80 = 3 of 4 (r4 misses)
    assert m["coverage_80"] == pytest.approx(0.75)
    # widths exp(3), exp(6), exp(1), exp(5); sorted e^1,e^3,e^5,e^6; median = (e^3 + e^5) / 2
    assert m["median_width_x"] == pytest.approx((math.exp(3) + math.exp(5)) / 2)
    # r1 d=[1,-.5,-2]: .1+.25+.2=.55; r2 d=[2,-1,-4]: .2+.5+.4=1.1; r3 d=[.5,0,-.5]: .05+0+.05=.1;
    # r4 d=[6,4,1]: .6+2+.9=3.5; pinball = (.55+1.1+.1+3.5) / (4*3) = 5.25 / 12 = 0.4375
    assert m["pinball"] == pytest.approx(0.4375)


# Score, 4 rows (true, [q10, q50, q90], weight):
#   r1 y=6 [5, 6.5, 8]   w=1  err .5, covered, width 3
#   r2 y=7 [6, 7, 8]     w=1  err 0,  covered, width 2
#   r3 y=8 [6, 7, 7.5]   w=2  err 1,  not covered (8>7.5), width 1.5
#   r4 y=5 [4, 6, 6.5]   w=4  err 1,  covered, width 2.5
SC_PRED = np.array([[5, 6.5, 8], [6, 7, 8], [6, 7, 7.5], [4, 6, 6.5]], dtype=float)
SC_TRUE = np.array([6, 7, 8, 5], dtype=float)
SC_W = np.array([1, 1, 2, 4], dtype=float)


def test_score_metrics_hand_worked():
    m = score_metrics(SC_PRED, SC_TRUE, SC_W)
    # weighted mae = (1*.5 + 1*0 + 2*1 + 4*1) / 8 = 6.5 / 8 = 0.8125
    assert m["mae"] == pytest.approx(0.8125)
    # coverage is unweighted: 3 of 4
    assert m["coverage_80"] == pytest.approx(0.75)
    # widths 3, 2, 1.5, 2.5; sorted 1.5, 2, 2.5, 3; median = 2.25
    assert m["median_width"] == pytest.approx(2.25)
    # row sums: r1 d=[1,-.5,-2] .1+.25+.2=.55; r2 d=[1,0,-1] .1+0+.1=.2;
    # r3 d=[2,1,.5] .2+.5+.45=1.15; r4 d=[1,-1,-1.5] .1+.5+.15=.75
    # weighted pinball = (1*.55 + 1*.2 + 2*1.15 + 4*.75) / (3 * 8) = 6.05 / 24
    assert m["pinball"] == pytest.approx(6.05 / 24)


def test_exactly_2x_counts_within():
    true = np.array([10.0, 10.0])
    pred = np.array([[9, 10 + math.log(2), 11], [9, 10 + 0.7, 11]])  # exactly 2x; just over 2x
    assert revenue_metrics(pred, true)["within_2x"] == pytest.approx(0.5)


def test_weights_change_score_mae():
    unweighted = score_metrics(SC_PRED, SC_TRUE, np.ones(4))["mae"]
    weighted = score_metrics(SC_PRED, SC_TRUE, SC_W)["mae"]
    assert unweighted == pytest.approx(2.5 / 4)  # (.5+0+1+1)/4
    assert weighted == pytest.approx(0.8125)
    assert weighted != pytest.approx(unweighted)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        revenue_metrics(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError):
        score_metrics(np.empty((0, 3)), np.empty(0), np.empty(0))
