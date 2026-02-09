
import pypdf

pdf_file = "Paper 2.pdf"
base_path = r"c:\Users\ponna\OneDrive\Desktop\Precog Task"
full_path = f"{base_path}\\{pdf_file}"

print(f"--- Attempting to extract from {pdf_file} ---")
try:
    reader = pypdf.PdfReader(full_path)
    print(f"Number of pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages[:5]):
        text = page.extract_text()
        print(f"--- Page {i+1} ---")
        print(text[:500]) # First 500 chars
        print("-" * 20)
except Exception as e:
    print(f"Error: {e}")
