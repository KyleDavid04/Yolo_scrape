```python
import pandas as pd

def append_to_csv(latex_formula, csv_file='latex_formulas.csv'):
    # Check if the CSV file already exists
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['LaTeX'])

    # Append the new LaTeX formula to the DataFrame
    new_row = {'LaTeX': latex_formula}
    df = df.append(new_row, ignore_index=True)

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
```