from src.utils import assign_user
def test_assign_stability():
    a = assign_user(12345, pct_treatment=0.5)
    b = assign_user(12345, pct_treatment=0.5)
    assert a == b
