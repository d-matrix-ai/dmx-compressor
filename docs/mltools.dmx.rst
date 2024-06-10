mltools.dmx package
===================

Submodules
----------

mltools.dmx.nn.Linear
----------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name                    target                      args                                                          kwargs
   -------------  ----------------------  --------------------------  ------------------------------------------------------------  --------
   placeholder    _input                  _input                      ()                                                            {}
   get_attr       input_cast_scale        input_cast.scale            ()                                                            {}
   get_attr       input_cast_zero_point   input_cast.zero_point       ()                                                            {}
   call_function  quantize                dmx.quantize                (_input, input_cast_scale, input_cast_zero_point, 'SAME')     {}
   call_function  dequantize              dmx.dequantize              (quantize, input_cast_scale, input_cast_zero_point)           {}
   get_attr       _weight                 _weight                     ()                                                            {}
   get_attr       weight_scale            weight_scale                ()                                                            {}
   get_attr       weight_zero_point       weight_zero_point           ()                                                            {}
   call_function  quantize_1              dmx.quantize                (_weight, weight_scale, weight_zero_point, 'SAME')            {}
   call_function  dequantize_1            dmx.dequantize              (quantize_1, weight_scale, weight_zero_point)                 {}
   get_attr       _bias                   _bias                       ()                                                            {}
   get_attr       bias_cast_scale         bias_cast.scale             ()                                                            {}
   get_attr       bias_cast_zero_point    bias_cast.zero_point        ()                                                            {}
   call_function  quantize_2              dmx.quantize                (_bias, bias_cast_scale, bias_cast_zero_point, 'SAME')        {}
   call_function  dequantize_2            dmx.dequantize              (quantize_2, bias_cast_scale, bias_cast_zero_point)           {}
   call_function  _output                 <built-in function linear>  (dequantize, dequantize_1, dequantize_2)                      {}
   get_attr       output_cast_scale       output_cast.scale           ()                                                            {}
   get_attr       output_cast_zero_point  output_cast.zero_point      ()                                                            {}
   call_function  quantize_3              dmx.quantize                (_output, output_cast_scale, output_cast_zero_point, 'SAME')  {}
   call_function  dequantize_3            dmx.dequantize              (quantize_3, output_cast_scale, output_cast_zero_point)       {}
   output         output                  output                      (dequantize_3,)                                               {}


mltools.dmx.nn.ResAdd
----------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name                      target                                                  args                                                               kwargs
   -------------  ------------------------  ------------------------------------------------------  -----------------------------------------------------------------  --------
   placeholder    _input                    _input                                                  ()                                                                 {}
   get_attr       input_cast_scale          input_cast.scale                                        ()                                                                 {}
   get_attr       input_cast_zero_point     input_cast.zero_point                                   ()                                                                 {}
   call_function  quantize                  dmx.quantize                                            (_input, input_cast_scale, input_cast_zero_point, 'SAME')          {}
   call_function  dequantize                dmx.dequantize                                          (quantize, input_cast_scale, input_cast_zero_point)                {}
   placeholder    residual                  residual                                                ()                                                                 {}
   get_attr       residual_cast_scale       residual_cast.scale                                     ()                                                                 {}
   get_attr       residual_cast_zero_point  residual_cast.zero_point                                ()                                                                 {}
   call_function  quantize_1                dmx.quantize                                            (residual, residual_cast_scale, residual_cast_zero_point, 'SAME')  {}
   call_function  dequantize_1              dmx.dequantize                                          (quantize_1, residual_cast_scale, residual_cast_zero_point)        {}
   call_function  output                    <built-in method add of type object at 0x7f9ef8059840>  (dequantize, dequantize_1)                                         {}
   get_attr       output_cast_scale         output_cast.scale                                       ()                                                                 {}
   get_attr       output_cast_zero_point    output_cast.zero_point                                  ()                                                                 {}
   call_function  quantize_2                dmx.quantize                                            (output, output_cast_scale, output_cast_zero_point, 'SAME')        {}
   call_function  dequantize_2              dmx.dequantize                                          (quantize_2, output_cast_scale, output_cast_zero_point)            {}
   output         output_1                  output                                                  (dequantize_2,)                                                    {}

mltools.dmx.nn.ActActMatMul
----------------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name                        target                                                     args                                                                     kwargs
   -------------  --------------------------  ---------------------------------------------------------  -----------------------------------------------------------------------  --------
   placeholder    _input                      _input                                                     ()                                                                       {}
   get_attr       input_cast_scale            input_cast.scale                                           ()                                                                       {}
   get_attr       input_cast_zero_point       input_cast.zero_point                                      ()                                                                       {}
   call_function  quantize                    dmx.quantize                                               (_input, input_cast_scale, input_cast_zero_point, 'SAME')                {}
   call_function  dequantize                  dmx.dequantize                                             (quantize, input_cast_scale, input_cast_zero_point)                      {}
   placeholder    multiplier                  multiplier                                                 ()                                                                       {}
   get_attr       multiplier_cast_scale       multiplier_cast.scale                                      ()                                                                       {}
   get_attr       multiplier_cast_zero_point  multiplier_cast.zero_point                                 ()                                                                       {}
   call_function  quantize_1                  dmx.quantize                                               (multiplier, multiplier_cast_scale, multiplier_cast_zero_point, 'SAME')  {}
   call_function  dequantize_1                dmx.dequantize                                             (quantize_1, multiplier_cast_scale, multiplier_cast_zero_point)          {}
   call_function  output                      <built-in method matmul of type object at 0x7f9ef8059840>  (dequantize, dequantize_1)                                               {}
   get_attr       output_cast_scale           output_cast.scale                                          ()                                                                       {}
   get_attr       output_cast_zero_point      output_cast.zero_point                                     ()                                                                       {}
   call_function  quantize_2                  dmx.quantize                                               (output, output_cast_scale, output_cast_zero_point, 'SAME')              {}
   call_function  dequantize_2                dmx.dequantize                                             (quantize_2, output_cast_scale, output_cast_zero_point)                  {}
   output         output_1                    output                                                     (dequantize_2,)                                                          {}


mltools.dmx.nn.Embedding
-------------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name       target                                  args               kwargs
   -------------  ---------  --------------------------------------  -----------------  -------------------------------------------------------------------------------------------------------
   placeholder    input_1    input                                   ()                 {}
   get_attr       weight     weight                                  ()                 {}
   call_function  embedding  <function embedding at 0x7f9e211aadd0>  (input_1, weight)  {'padding_idx': None, 'max_norm': None, 'norm_type': 2.0, 'scale_grad_by_freq': False, 'sparse': False}
   output         output     output                                  (embedding,)       {}

mltools.dmx.nn.SoftMax 
-----------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name     target                                args        kwargs
   -------------  -------  ------------------------------------  ----------  --------
   placeholder    _input   _input                                ()          {}
   call_function  softmax  <function softmax at 0x7f9e211aa7a0>  (_input,)   {}
   output         output   output                                (softmax,)  {}

mltools.dmx.nn.LayerNorm 
-------------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name                  target                                   args                                                     kwargs
   -------------  --------------------  ---------------------------------------  -------------------------------------------------------  --------
   placeholder    _input                _input                                   ()                                                       {}
   get_attr       _weight               _weight                                  ()                                                       {}
   get_attr       weight_scale          weight_scale                             ()                                                       {}
   get_attr       weight_zero_point     weight_zero_point                        ()                                                       {}
   call_function  quantize              dmx.quantize                             (_weight, weight_scale, weight_zero_point, 'SAME')       {}
   call_function  dequantize            dmx.dequantize                           (quantize, weight_scale, weight_zero_point)              {}
   get_attr       _bias                 _bias                                    ()                                                       {}
   get_attr       bias_cast_scale       bias_cast.scale                          ()                                                       {}
   get_attr       bias_cast_zero_point  bias_cast.zero_point                     ()                                                       {}
   call_function  quantize_1            dmx.quantize                             (_bias, bias_cast_scale, bias_cast_zero_point, 'SAME')   {}
   call_function  dequantize_1          dmx.dequantize                           (quantize_1, bias_cast_scale, bias_cast_zero_point)      {}
   get_attr       normalized_shape      normalized_shape                         ()                                                       {}
   get_attr       eps                   eps                                      ()                                                       {}
   call_function  ln                    <function layer_norm at 0x7f9e211ab1c0>  (_input, normalized_shape, quantize, dequantize_1, eps)  {}
   output         output                output                                   (ln,)                                                    {}

mltools.dmx.nn.Dropout 
-----------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name     target                                args        kwargs
   -------------  -------  ------------------------------------  ----------  ----------------------------------------------
   placeholder    input_1  input                                 ()          {}
   call_function  dropout  <function dropout at 0x7f9e211a9bd0>  (input_1,)  {'p': 0.5, 'training': True, 'inplace': False}
   output         output   output                                (dropout,)  {}

mltools.dmx.nn.GELU 
--------------------------
**to_compiler_graph**

.. code-block:: 

   opcode         name     target                    args        kwargs
   -------------  -------  ------------------------  ----------  -----------------------
   placeholder    input_1  input                     ()          {}
   call_function  gelu     <built-in function gelu>  (input_1,)  {'approximate': 'none'}
   output         output   output                    (gelu,)     {}


