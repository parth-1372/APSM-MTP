import os, glob, re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We only care about models.py, event.py, history.py, aggregators.py, simulator.py, utils/metrics.py
    if 'Optional' not in content:
        content = content.replace('from typing import ', 'from typing import Optional, ')
    
    new_content = re.sub(r'(\w+(?:\[[^\]]+\])?)\s*\|\s*None', r'Optional[\1]', content)
    
    with open(filepath, 'w') as f:
        f.write(new_content)

for p in glob.glob('src/**/*.py', recursive=True):
    fix_file(p)
    print(f"Fixed {p}")
