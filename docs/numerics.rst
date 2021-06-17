Corsair numerics
================


Shorthands
----------

We use string shorthands to specify numerical formats. 

For example, one can instantiate a format by:

.. code-block:: python

    from numerical import Format
    input_format = Format.from_shorthand("BFP[8|8]{64,-1}(N)")

This is equivalent to:

.. code-block:: python

    from numerical import BlockFloatingPoint
    input_format = BlockFloatingPoint(
        precision=8,
        block_size=64,
        block_dim=-1,
        rounding="nearest",
    )

Shorthands are used in Corsair configuration YAML files for convenience, such as:

.. code-block :: yaml

    - &DUMMY_FORMAT                 SAME
    - &IMC_GEMM_FORMAT_HIGH         BFP[8|8]{64,-1}(N)
    - &IMC_GEMM_FORMAT_LOW          BFP[4|8]{128,-1}(N)
    - &IMC_CONV_FORMAT_HIGH         BFP[8|8]{64,1}(N)
    - &IMC_CONV_FORMAT_LOW          BFP[4|8]{128,1}(N)
    - &IMC_ACCUM_FORMAT_HIGH        FP[1|8|23](N)
    - &IMC_GEMM_ACCUM_FORMAT_LOW    BFP[22|8]{64,-1}(N)
    - &IMC_CONV_ACCUM_FORMAT_LOW    BFP[22|8]{64,1}(N)
    - &OB_FORMAT                    FP[1|8|23](N)
    - &SIMD_FORMAT_HIGH             FP[1|8|23](N)
    - &SIMD_FORMAT_LOW              XP[25,+12](CSN)

