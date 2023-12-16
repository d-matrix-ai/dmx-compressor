Methodology
-----------

We extend and integrate `QPyTorch`_ in this project.
All formats are simulated with single-precision floating point.  
Thus, the highest integer precision that can be realized is ``24``-bit, and the highest floating point dynamic range ``8``-bit.  

.. _QPyTorch: https://github.com/Tiiiger/QPyTorch

Format shorthands
-----------------

Fopr convenience, we use string shorthands to specify numerical formats. 

For example, one can instantiate a format object by:

.. code-block:: python

    from numerical import Format
    input_format = Format.from_shorthand("BFP[8|8]{64,-1}(SN)")

This is equivalent to:

.. code-block:: python

    from numerical import BlockFloatingPoint
    input_format = BlockFloatingPoint(
        precision=8,
        block_size=64,
        block_dim=-1,
        symmetric=True,
        rounding="nearest",
    )

Shorthand strings are composed of 4 parts:

.. code-block:: 

    IDENTIFIER[element_spec]{tensor_spec}(cast_behavior)

Only the first part, i.e. the identifier, is required, the rest being conditional upon specific formats.  

Same
~~~~

This is a dummy format, cast into this format is a no-op.  

The shorthand is::

    SAME

Floating point
~~~~~~~~~~~~~~

This is a floating point format, with each element having an optional sign bit ``s`` (``1`` for signed and ``0`` for unsigned), a ``m``-bit mantissa, an ``e``-bit exponent and an exponent bias ``b``.  

Two casting behavior are supported: 

* ``X``: flush submornals, which is ``F`` for flushing, ``_`` for not flushing.
* ``Y``: rounding mode, which is ``N`` for nearest (even when tied), ``S`` for stochastic rounding.  

The shorthand is::

    FP[s|e|m,b](XY)

Block floating point
~~~~~~~~~~~~~~~~~~~~

This is a block floating point format, with each element having a ``n``-bit signed integer significand and an ``8``-bit shared exponent.  

Blocks are groups of ``b`` contiguous elements along tensor dimension ``d``.

One casting behavior is supported: 

* ``X``: rounding mode, which is ``N`` for nearest, ``S`` for stochastic rounding.  

The shorthand is::

    BFP[n|8]{b,d}(X)

Fixed point
~~~~~~~~~~~

This is a fixed point format, with each element having a ``n``-bit signed integer significand. 
Position of the radix point is specified by a bias of ``±b``-bit shift.

Three casting behavior ``XYZ`` are supported, in exact order as follows: 

* ``X``: clamping of out-of-range numbers, which is ``C`` for clamp, ``U`` for unclamp.
* ``Y``: symmetric/asymmetric quantization range, which is ``S`` for symmetric, ``A`` for asymmetric.
* ``Z``: rounding mode, which is ``N`` for nearest, ``S`` for stochastic rounding.  

The shorthand is::

    XP[n,±b](XYZ)
