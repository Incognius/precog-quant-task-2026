
import os
import sys
import pypdf

pdf_files = [
    "Visibility Graphs.pdf",
    "Paper 2.pdf",
    "Time_Series_Analysis_Based_on_Visibility_Graph_Theory.pdf"
]

base_path = r"c:\Users\ponna\OneDrive\Desktop\Precog Task"
output_file = os.path.join(base_path, "pdf_text_utf8.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for pdf_file in pdf_files:
        f.write(f"--- Extracting from {pdf_file} ---\n")
        try:
            reader = pypdf.PdfReader(os.path.join(base_path, pdf_file))
            pages_to_read = min(3, len(reader.pages))
            for i in range(pages_to_read):
                f.write(f"--- Page {i+1} ---\n")
                text = reader.pages[i].extract_text()
                f.write(text + "\n")
            
            if len(reader.pages) > 3:
                 f.write(f"--- Last Page ---\n")
                 text = reader.pages[-1].extract_text()
                 f.write(text + "\n")
                 
        except Exception as e:
            f.write(f"Error reading {pdf_file}: {e}\n")
        f.write("\n" + "="*50 + "\n\n")

print(f"Text extracted to {output_file}")
