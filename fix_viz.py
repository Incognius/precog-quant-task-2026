import json

# Read notebook
with open('notebooks/Final_Notebooks/Stage3_v2_Backtesting.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the visualization cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'LONG-ONLY VISUALIZATION' in ''.join(cell['source']):
        # Replace the problematic line
        new_source = []
        for line in cell['source']:
            if 'calculate_metrics(pd.DataFrame({\"date\": ew_oos.index' in line:
                # Replace with simple sharpe calculation
                new_source.append("ax.plot(dates_plot, ew_cum.values, label=f'EW Benchmark (Sharpe={ew_sharpe:.2f})', \n")
            elif 'ew_cum = (1 + ew_oos).cumprod()' in line:
                new_source.append(line)
                # Add the simple sharpe calculation
                new_source.append("\n")
                new_source.append("# Simple Sharpe calculation for benchmark\n")
                new_source.append("ew_sharpe = ew_oos.mean() / ew_oos.std() * np.sqrt(252)\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        print("Fixed visualization cell")
        break

# Write back
with open('notebooks/Final_Notebooks/Stage3_v2_Backtesting.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done")
