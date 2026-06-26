import urllib.request
import zipfile
import os
import sys

url = "https://zenodo.org/api/records/15393791/files/data.zip/content"
filename = "data.zip"

print(f"Downloading {filename}...")
# Remove incomplete file if exists
if os.path.exists(filename):
    os.remove(filename)

urllib.request.urlretrieve(url, filename)
print("Download complete.")

print("Extracting...")
os.makedirs('src', exist_ok=True)
with zipfile.ZipFile(filename, 'r') as z:
    z.extractall('src')
print("Extraction complete.")
