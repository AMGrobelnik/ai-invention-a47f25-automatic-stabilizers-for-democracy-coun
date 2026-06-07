import fitz
import os

doc = fitz.open("paper.pdf")
os.makedirs("pages", exist_ok=True)
for i, page in enumerate(doc):
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(f"pages/page_{i+1:02d}.png")
print(f"Converted {len(doc)} pages")
