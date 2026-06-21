import fitz
doc = fitz.open("knowledge_base/Floods_Dos_and_Donts.pdf")
print(doc[0].get_text()[:500])