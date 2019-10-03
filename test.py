from model import run_model


def test_model_performance():
    results = run_model()
    assert results < 20
