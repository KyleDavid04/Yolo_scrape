```python
import pytesseract
from sympy import *
from sympy.parsing.sympy_parser import (parse_expr,
standard_transformations, implicit_multiplication_application)

# Set up transformations for parsing
transformations = (standard_transformations +
(implicit_multiplication_application,))

def convert_to_latex(formula_text):
    """
    Convert the extracted text to LaTeX.
    """
    try:
        # Parse the formula using sympy
        formula = parse_expr(formula_text, transformations=transformations)
        
        # Convert the formula to LaTeX
        latex_formula = latex(formula)
        
        return latex_formula
    except Exception as e:
        print(f"Error in converting to LaTeX: {str(e)}")
        return None
```