import zipfile
import os

os.makedirs('src', exist_ok=True)
try:
    with zipfile.ZipFile('data.zip', 'r') as z:
        z.extractall('src')
    print("Successfully extracted data.zip to src/")
except Exception as e:
    print(f"Extraction failed: {e}")
