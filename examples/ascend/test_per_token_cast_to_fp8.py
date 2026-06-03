from example_simdvf_per_token_cast_to_fp8 import test as _run_simdvf_correctness
from example_simtvf_per_token_cast_to_fp8 import test as _run_simtvf_correctness


TEST_M = 8192
TEST_N = 8192


def test_simdvf_per_token_cast_to_fp8():
    _run_simdvf_correctness(TEST_M, TEST_N, print_source=False)


def test_simtvf_per_token_cast_to_fp8():
    _run_simtvf_correctness(TEST_M, TEST_N, print_source=False)


def main():
    test_simdvf_per_token_cast_to_fp8()
    test_simtvf_per_token_cast_to_fp8()


if __name__ == "__main__":
    main()
