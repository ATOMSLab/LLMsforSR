import json
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mse(equation, data, fitted_params):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        y = data[:, num_indep_vars]
        predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

        if np.isscalar(predicted_y):
            predicted_y = np.full(y.shape, predicted_y)
        mse = mean_squared_error(y, predicted_y)
        return round(mse, 8)

def calculate_mae(equation, data, fitted_params):
        num_indep_vars = data.shape[1] - 1
        x = data[:, :num_indep_vars]
        y = data[:, num_indep_vars]
        predicted_y = eval(equation, {'c': fitted_params, 'np': np,'sqrt': np.sqrt, 'cbrt': np.cbrt, 'log': np.log, 'exp': np.exp, **{f'x{i+1}':x[:,i].reshape(-1) for i in range(num_indep_vars)}})

        if np.isscalar(predicted_y):
            predicted_y = np.full(y.shape, predicted_y)
        mae = mean_absolute_error(y, predicted_y)
        return round(mae, 8)

def calculate_complexity(equation):
    binary_operator_pattern = re.compile(r'\*{2}|\*{1}|[+]|[-]|[/]')
    unary_function_pattern = re.compile(r'\b(log|exp|sqrt|sin|cos|tan|abs)\b')

    binary_operators = len(binary_operator_pattern.findall(equation))
    unary_functions = len(unary_function_pattern.findall(equation))
    
    complexity = (binary_operators * 2) + (unary_functions * 1) + 1
    
    return complexity

def custom_sorting(data):
        if isinstance(data, str):
            data = json.loads(data)
        
        if len(data) < 6:
            return data        
        else:
            new_data = data.copy()
            new_data = sorted(new_data,key=lambda x: (x['mse'], x['complexity']), reverse=True)
            drop = 0
            i = 0
            while drop < 3 and i < len(new_data):
                current_entry = new_data[i]
                for j in range(i + 1, len(new_data)):
                    if new_data[j]['complexity'] == current_entry['complexity']:
                        new_data.pop(i)
                        drop = drop + 1
                        break
                else:
                    i += 1
            return new_data
        

def format_and_parse_expressions(expression_string):
        expression_string = expression_string.strip()
        if expression_string.startswith('[') and expression_string.endswith(']'):
            expression_string = expression_string[1:-1].strip()

        if '\n' in expression_string:
            lines = [line.strip() for line in expression_string.split('\n') if line.strip()]
        else:
            lines = [line.strip() for line in expression_string.split(',') if line.strip()]

        parsed_expressions = []
        for line in lines:
            line = line.strip('"')

            line = line.strip().strip('"').strip()
            line = line.rstrip(',').strip('"')
            if not line:
                continue
        
        # Remove leading numbers (like '1.', '2.') from the line
            line = re.sub(r'^\d+\.\s*', '', line).strip()

        # Replace c_{0}, c_0, c_{1}, c_1, etc. with c[0], c[1], etc.
            line = re.sub(r'c_\{(\d+)\}', r'c[\1]', line)  # Handle c_{0} notation
            line = re.sub(r'c_(\d+)', r'c[\1]', line)      # Handle c_0 notation

        # Handle x1_1, x2_2, x_1, x_2, etc. by converting them to x1, x2, etc.
            line = re.sub(r'([a-zA-Z]\d?)_(\d+)', r'\1', line)  # Replace x1_1 with x1, x_1 with x1, etc.
            parsed_expressions.append(line)

        return parsed_expressions


def format_expressions(expressions):
        formatted_expressions = []

        for expression in expressions:
            # Split the expression into left (variable) and right (formula) parts
            if "=" in expression:
                _, formula = expression.split("=")  # We are discarding the left side
                formula = formula.strip()
            else:
                formula = expression.strip()

            # Process the formula (right side)
            formula = re.sub(r"\\sqrt\{([^}]+)\}", r"(\1)**0.5", formula)  # Replace \sqrt{content} with content**0.5
            formula = re.sub(r"\\cbrt\{([^}]+)\}", r"(\1)**(1/3)", formula)  # Replace \cbrt{content} with content**(1/3)
            formula = formula.replace(r"cube\_root", "**(1/3)" )
            formula = formula.replace(r"\log", "log").replace(r"\exp", "exp")
            formula = formula.replace(r"\\log", "log").replace(r"\\exp", "exp")
            formula = formula.replace(r"log10", "log")
            formula = re.sub(r'cube_root\(([^)]+)\)', r'\1**(1/3)', formula)
            formula = re.sub(r'cubert\(([^)]+)\)', r'\1**(1/3)', formula)
            formula = re.sub(r'cube\(([^)]+)\)', r'\1**3', formula)
            formula = re.sub(r'square\(([^)]+)\)', r'\1**2', formula)
            formula = formula.replace("log10*", "log")
            formula = formula.replace("e**", "exp")
            formula = formula.replace("\\cdot", "*")
            formula = re.sub(r"c(\d+)", r"c[\1]", formula)  # Replace c0, c1, etc. with c[0], c[1], etc.
            formula = re.sub(r"\{([^}]+)\}", r"(\1)", formula) # Replace { } with ( )
            formula = formula.replace("^", "**")  # Replace ^ with **
            formula = formula.replace(" ", "")  # Remove white space
            formula = re.sub(r"(?<![a-zA-Z])x(?![a-zA-Z0-9])", "x1", formula)  # Replace x with x1 if it's not followed by a digit
            formula = re.sub(r"(?<![a-zA-Z])y(?![a-zA-Z0-9])", "x2", formula)  # Replace y with x2 if it's not followed by a digit
            formula = re.sub(r"(?<![a-zA-Z])z(?![a-zA-Z0-9])", "x3", formula)  # Replace z with x3 if it's not followed by a digit
            formula = formula.replace("$", "") #Replace $ signs if present
            formula = formula.replace(')(', ')*(')
            formula  = formula.replace('\\frac', '').replace(')(', ')/(') if 'frac' in formula else formula
            formula  = formula.replace('frac', '').replace(')(', ')/(') if 'frac' in formula else formula
      
            formatted_expressions.append(formula)

        return formatted_expressions
