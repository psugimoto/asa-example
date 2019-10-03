from model import run_model

# To run tests: python -m pytest test.py


def test_model_performance():
    results = run_model()
    assert results < 20
