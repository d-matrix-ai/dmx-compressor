Corsair numerics
================

We employ `QPyTorch`_ to realize numerical capabilities of Corsair.
All formats are simulated with single-precision floating point.  
Thus, the highest integer precision that can be realized is ``24``, and the highest floating point dynamic range ``8``-bit.  

.. _QPyTorch: https://github.com/Tiiiger/QPyTorch

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

.. code-block:: yaml

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

Shorthand strings are composed of four parts:

.. code-block:: 

    IDENTIFIER[element_bit_map]{tensor_spec}(cast_behavior)

Only the first part, i.e. the identifier, is required, the rest being conditional upon specific formats.  

Same
~~~~

This is a dummy format, cast into this format is a no-op.  

The shorthand is::

    SAME

Floating point
~~~~~~~~~~~~~~

This is a floating point format, with each element having a sign bit, a ``m``-bit mantissa and an ``e``-bit exponent.  

One casting behavior ``B`` is supported: 

* Rounding mode, which is ``N`` for nearest, ``S`` for stochastic rounding.  

The shorthand is::

    FP[1|e|m](B)

Block floating point
~~~~~~~~~~~~~~~~~~~~

This is a block floating point format, with each element having a ``n``-bit signed integer significand and an ``8``-bit shared exponent.  

Blocks are groups of ``b`` contiguous elements along tensor dimension ``d``.

One casting behavior ``B`` is supported: 

* Rounding mode, which is ``N`` for nearest, ``S`` for stochastic rounding.  

The shorthand is::

    BFP[n|8]{b,d}(B)

Fixed point
~~~~~~~~~~~

This is a fixed point format, with each element having a ``n``-bit signed integer significand. 
Position of the radix point is specified by a bias of ``±b``-bit shift.

Three casting behavior ``BBB`` are supported, in exact order as follows: 

* Clamping of out-of-range numbers, which is ``C`` for clamp, ``U`` for unclamp.
* Symmetric/asymmetric quantization range, which is ``S`` for symmetric, ``A`` for asymmetric.
* Rounding mode, which is ``N`` for nearest, ``S`` for stochastic rounding.  

The shorthand is::

    XP[n,±b](BBB)
