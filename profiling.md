## with broadcast_to:

 8468643 function calls (8468641 primitive calls) in 11.019 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       81    0.085    0.001   19.187    0.237 base_events.py:1976(_run_once)
    60000    0.140    0.000   10.616    0.000 1856880170.py:198(step_accumulate_gradients)
    60000    0.189    0.000    7.012    0.000 1856880170.py:160(accumulate_gradients)
   240000    3.212    0.000    6.831    0.000 1856880170.py:81(accumulate_gradients)
       80    0.002    0.000    5.967    0.075 events.py:92(_run)
   480000    0.233    0.000    3.554    0.000 _stride_tricks_impl.py:474(broadcast_to)
   480000    2.285    0.000    3.321    0.000 _stride_tricks_impl.py:447(_broadcast_to)
      2/1    0.000    0.000    1.538    1.538 {built-in method builtins.exec}
      2/1    0.000    0.000    1.538    1.538 <string>:1(<module>)
    60000    0.167    0.000    1.471    0.000 1856880170.py:150(__call__)
   240000    1.304    0.000    1.304    0.000 1856880170.py:73(__call__)
    60000    0.444    0.000    1.159    0.000 1856880170.py:221(z_loss)
    60000    0.362    0.000    0.837    0.000 1856880170.py:214(z_loss_gradient)
   300000    0.412    0.000    0.798    0.000 fromnumeric.py:66(_wrapreduction)
       81    0.003    0.000    0.664    0.008 selectors.py:310(select)
   180000    0.180    0.000    0.663    0.000 fromnumeric.py:2342(sum)
   120000    0.098    0.000    0.461    0.000 fromnumeric.py:3049(max)
   300000    0.312    0.000    0.312    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   480000    0.176    0.000    0.270    0.000 _function_base_impl.py:374(iterable)
       80    0.002    0.000    0.268    0.003 {method 'run' of '_contextvars.Context' objects}
...
        1    0.000    0.000    0.000    0.000 {built-in method _thread.allocate_lock}
        1    0.000    0.000    0.000    0.000 {method 'acquire' of '_thread.RLock' objects}
        1    0.000    0.000    0.000    0.000 <string>:2(__init__)
        1    0.000    0.000    0.000    0.000 {method 'release' of '_thread.RLock' objects}
        1    0.000    0.000    0.000    0.000 {method 'release' of '_thread.lock' objects}

## with np.outer:

         4148547 function calls (4148543 primitive calls) in 7.404 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       78    0.078    0.001   14.546    0.186 base_events.py:1976(_run_once)
    60000    0.137    0.000    7.035    0.000 1707987795.py:196(step_accumulate_gradients)
      2/1    0.000    0.000    5.124    5.124 {built-in method builtins.exec}
        1    0.002    0.002    4.541    4.541 <string>:1(<module>)
    60000    0.163    0.000    3.568    0.000 1707987795.py:158(accumulate_gradients)
   240000    1.624    0.000    3.409    0.000 1707987795.py:81(accumulate_gradients)
       75    0.001    0.000    2.320    0.031 events.py:92(_run)
   240000    1.522    0.000    1.750    0.000 numeric.py:905(outer)
    60000    0.160    0.000    1.436    0.000 1707987795.py:148(__call__)
   240000    1.277    0.000    1.277    0.000 1707987795.py:73(__call__)
    60000    0.424    0.000    1.089    0.000 1707987795.py:219(z_loss)
    60000    0.348    0.000    0.804    0.000 1707987795.py:212(z_loss_gradient)
   300000    0.387    0.000    0.760    0.000 fromnumeric.py:66(_wrapreduction)
   180000    0.167    0.000    0.635    0.000 fromnumeric.py:2342(sum)
   120000    0.093    0.000    0.426    0.000 fromnumeric.py:3049(max)
   300000    0.299    0.000    0.299    0.000 {method 'reduce' of 'numpy.ufunc' objects}
     5000    0.147    0.000    0.244    0.000 1707987795.py:202(step_learn)
   480000    0.137    0.000    0.137    0.000 {method 'ravel' of 'numpy.ndarray' objects}
   480000    0.091    0.000    0.091    0.000 {built-in method numpy.asarray}
        1    0.001    0.001    0.090    0.090 1707987795.py:14(train)

## using z_loss_and_gradient instead of 2 functions calculating loss seperately:

        2888599 function calls (2888598 primitive calls) in 6.176 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       81    0.064    0.001    6.433    0.079 base_events.py:1976(_run_once)
    60000    0.107    0.000    5.813    0.000 1190115785.py:196(step_accumulate_gradients)
      2/1    0.000    0.000    4.727    4.727 {built-in method builtins.exec}
        1    0.000    0.000    3.760    3.760 <string>:1(<module>)
    60000    0.156    0.000    3.522    0.000 1190115785.py:158(accumulate_gradients)
   240000    1.607    0.000    3.369    0.000 1190115785.py:81(accumulate_gradients)
   240000    1.508    0.000    1.727    0.000 numeric.py:905(outer)
       77    0.001    0.000    1.487    0.019 events.py:92(_run)
    60000    0.155    0.000    1.411    0.000 1190115785.py:148(__call__)
   240000    1.256    0.000    1.256    0.000 1190115785.py:73(__call__)
       81    0.002    0.000    0.975    0.012 selectors.py:310(select)
    60000    0.436    0.000    0.775    0.000 1190115785.py:214(z_loss_and_gradient)
        1    0.006    0.006    0.530    0.530 1190115785.py:14(train)
       77    0.006    0.000    0.528    0.007 {method 'run' of '_contextvars.Context' objects}
     5000    0.148    0.000    0.241    0.000 1190115785.py:204(step_learn)
   120000    0.061    0.000    0.210    0.000 {method 'sum' of 'numpy.ndarray' objects}
       81    0.003    0.000    0.207    0.003 selectors.py:304(_select)
   180000    0.178    0.000    0.178    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   120000    0.041    0.000    0.148    0.000 _methods.py:47(_sum)
   480000    0.132    0.000    0.132    0.000 {method 'ravel' of 'numpy.ndarray' objects}
...
        1    0.000    0.000    0.000    0.000 history.py:1225(hold)
        1    0.000    0.000    0.000    0.000 {method '__enter__' of 'sqlite3.Connection' objects}
        1    0.000    0.000    0.000    0.000 <string>:2(__init__)
        1    0.000    0.000    0.000    0.000 base_events.py:1971(_timer_handle_cancelled)
        1    0.000    0.000    0.000    0.000 {method 'release' of '_thread.lock' objects}

## got rid of np.outer using `np.multiply(gradients_forward[:, np.newaxis], self.activationsin, out=self._tmp)` now

        1444690 function calls (1444685 primitive calls) in 5.290 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       53    0.063    0.001    9.673    0.183 base_events.py:1976(_run_once)
    60000    0.100    0.000    4.954    0.000 3057037471.py:196(step_accumulate_gradients)
       52    0.004    0.000    4.895    0.094 events.py:92(_run)
    60000    0.149    0.000    2.741    0.000 3057037471.py:158(accumulate_gradients)
   240000    2.594    0.000    2.594    0.000 3057037471.py:81(accumulate_gradients)
    60000    0.149    0.000    1.379    0.000 3057037471.py:148(__call__)
   240000    1.229    0.000    1.229    0.000 3057037471.py:73(__call__)
    60000    0.417    0.000    0.737    0.000 3057037471.py:214(z_loss_and_gradient)
      2/1    0.000    0.000    0.416    0.416 {built-in method builtins.exec}
      2/1    0.000    0.000    0.416    0.416 <string>:1(<module>)
        1    0.005    0.005    0.416    0.416 3057037471.py:14(train)
     5000    0.139    0.000    0.229    0.000 3057037471.py:204(step_learn)
   120000    0.056    0.000    0.199    0.000 {method 'sum' of 'numpy.ndarray' objects}
   180000    0.170    0.000    0.170    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   120000    0.039    0.000    0.143    0.000 _methods.py:47(_sum)
    60000    0.034    0.000    0.121    0.000 {method 'max' of 'numpy.ndarray' objects}
    60000    0.022    0.000    0.088    0.000 _methods.py:39(_amax)
     5000    0.008    0.000    0.056    0.000 3057037471.py:33(zero_grad)
    20000    0.013    0.000    0.048    0.000 3057037471.py:102(zero_grad)
       52    0.000    0.000    0.039    0.001 {method 'run' of '_contextvars.Context' objects}
...
        1    0.000    0.000    0.000    0.000 {method '__enter__' of 'sqlite3.Connection' objects}
        2    0.000    0.000    0.000    0.000 {method 'release' of '_thread.lock' objects}
        1    0.000    0.000    0.000    0.000 {method 'release' of '_thread.RLock' objects}
        1    0.000    0.000    0.000    0.000 <string>:2(__init__)
        1    0.000    0.000    0.000    0.000 history.py:1225(hold)

## batching with batch size 12:

 373567 function calls (373565 primitive calls) in 1.250 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       15    0.012    0.001    1.663    0.111 base_events.py:1976(_run_once)
     5000    0.010    0.000    1.008    0.000 3363355070.py:253(step_accumulate_batch)
      2/1    0.000    0.000    0.793    0.793 {built-in method builtins.exec}
      2/1    0.000    0.000    0.793    0.793 <string>:1(<module>)
     5000    0.016    0.000    0.546    0.000 3363355070.py:202(accumulate_gradients_batch)
    20000    0.466    0.000    0.531    0.000 3363355070.py:105(accumulate_gradients_batch)
       15    0.000    0.000    0.463    0.031 events.py:92(_run)
     5000    0.011    0.000    0.326    0.000 3363355070.py:187(forward_batch)
    20000    0.316    0.000    0.316    0.000 3363355070.py:96(forward_batch)
     5000    0.133    0.000    0.218    0.000 3363355070.py:247(step_learn)
     5000    0.055    0.000    0.126    0.000 3363355070.py:261(z_loss_and_gradient_batch)
    40000    0.023    0.000    0.112    0.000 {method 'sum' of 'numpy.ndarray' objects}
    45000    0.094    0.000    0.094    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    40000    0.014    0.000    0.090    0.000 _methods.py:47(_sum)
     5000    0.007    0.000    0.053    0.000 3363355070.py:39(zero_grad)
    20000    0.013    0.000    0.045    0.000 3363355070.py:136(zero_grad)
        1    0.000    0.000    0.045    0.045 3363355070.py:14(train)
    40000    0.033    0.000    0.033    0.000 {method 'fill' of 'numpy.ndarray' objects}
    45000    0.022    0.000    0.032    0.000 3363355070.py:192(parameters)
     5000    0.004    0.000    0.024    0.000 {method 'max' of 'numpy.ndarray' objects}
...
        1    0.000    0.000    0.000    0.000 {built-in method _thread.allocate_lock}
        2    0.000    0.000    0.000    0.000 traitlets.py:3484(validate_elements)
        1    0.000    0.000    0.000    0.000 {method 'release' of '_thread.lock' objects}
        1    0.000    0.000    0.000    0.000 <string>:2(__init__)
        1    0.000    0.000    0.000    0.000 typing.py:2300(cast)