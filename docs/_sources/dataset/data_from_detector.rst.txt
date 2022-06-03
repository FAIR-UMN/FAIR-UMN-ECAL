Data from the Detector
========================

QPsolver
------------------

String in {'osqp'}. Default value: 'osqp'

Select the QP solver used in the steering strategy and termination condition. Currently only OSQP is supported.

torch_device
--------------------------------

torch.device('cpu') OR torch.device('cuda'). Default value: torch.device('cpu')

Choose torch.device used for matrix operation in PyGRANSO.
opts.torch_device = torch.device('cuda') if one wants to use cuda device.

globalAD
--------------------------------

Boolean value. Default value: True

Compute all gradients of objective and constraint functions via auto-differentiation.
In the default setting, user should provide [f,ci,ce] = combined_fn(X).
When globalAD = False, user should provide [f,f_grad,ci,ci_grad,ce,ce_grad] = combined_fn(X). 
Please check the docstring of pygranso.py for more details of setting combined_fn.

double_precision
--------------------------------

Boolean value. Default value: True

Set the floating number formats to be double precision for PyGRANSO solver. If double_precision = False, 
the floating number formats will be single precision.

 