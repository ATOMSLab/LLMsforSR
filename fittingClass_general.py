import numpy as np
import json
import re
from scipy import optimize as opt
import math
from helper import *

class FittingOptimizer:
    def __init__(self):
        self.equation_indices_pattern = re.compile(r'c\[(\d+)\]')

    
    def get_equation_indices(self, equation):
        return sorted([int(index) for index in self.equation_indices_pattern.findall(equation)], 
                      reverse=True)

    
    def equation_error(self, c, equation, data):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        return np.mean((eval(equation, {'c': c, 'np': np, 'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}}) - data[:, num_indep_vars])**2)


    def fitting_constants(self, indep_vars, dep_var, expressions):
        results = []
        # Format the expressions first
        expressions = format_expressions(expressions)
        
        # Reshape the inputs and stack the data
        data = np.column_stack([var.reshape(-1, 1) if isinstance(var, np.ndarray) else np.array(eval(var)).reshape(-1, 1) for var in indep_vars] + 
                               [dep_var.reshape(-1, 1) if isinstance(dep_var, np.ndarray) else np.array(eval(dep_var)).reshape(-1, 1)])
        
        # Parse expressions as JSON if it's a string
        if isinstance(expressions, str):
            expressions = json.loads(expressions)

        for equation in expressions:
            equation_indices = self.get_equation_indices(equation)
            constant = len(equation_indices)
            initial_val = [1] * constant

            undesired_patterns = [r"sin", r"cos", r"tan"]
            if any(re.search(pattern, equation) for pattern in undesired_patterns):
                result_dict = {'equation': equation, 'complexity': self.calculate_complexity(equation), 'mse': float('inf')}
                results.append(result_dict)
                continue


            # Check if equation indices are within the available parameters
            if all(index < constant for index in equation_indices):
                try:
                    result = opt.basinhopping(func=self.equation_error, x0=initial_val,
                                          minimizer_kwargs={"method": "Nelder-Mead", 
                                                            "args": (equation, data)})
                    fitted_params = result.x

                    mse = calculate_mse(equation, data, fitted_params)
                    complexity = calculate_complexity(equation)

                    result_dict = {'equation': equation, 'complexity': complexity, 'mse': mse}
                    results.append(result_dict)
                except (RuntimeError, ValueError) as e:
                    high_mse_value = float('inf') # Assigning a sufficiently large value
                    result_dict = {'equation': equation, 'complexity': calculate_complexity(equation), 'mse': high_mse_value}
                    results.append(result_dict)

        results.sort(key=lambda x: (x['mse'], x['complexity']),reverse=True)
        return json.dumps(results, indent=3)