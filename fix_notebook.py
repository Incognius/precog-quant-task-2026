content = open('notebooks/Final_Notebooks/Stage3_v2_Backtesting.ipynb', 'r', encoding='utf-8').read()
content = content.replace("results.append({'date': date, 'net_return': long_ret - tc})", "results.append({'date': date, 'net_return': long_ret - tc, 'turnover': turnover})")
open('notebooks/Final_Notebooks/Stage3_v2_Backtesting.ipynb', 'w', encoding='utf-8').write(content)
print('Fixed')
