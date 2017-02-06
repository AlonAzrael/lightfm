try:
    # Import OpenMP-enabled extension
    from ._mega_lightfm_fast import *  # NOQA
    from ._mega_lightfm_fast import __test_in_positives  # NOQA
except ImportError as e:
    # Fall back on OpenMP-less extension
    raise e
    import warnings

    warnings.warn('LightFM was compiled without OpenMP support. '
                  'Only a single thread will be used.')

    from ._lightfm_fast_no_openmp import *  # NOQA
    from ._lightfm_fast_no_openmp import __test_in_positives  # NOQA
