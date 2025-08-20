import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF.
    :param pdf_path: Ruta absoluta o relativa del PDF.
    :return: Texto completo extraído.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return ""

def process_pdfs_in_folder(folder_path: str, output_folder: str):
    """
    Procesa todos los PDFs de una carpeta, extrae su texto y genera un .txt por cada PDF.
    :param folder_path: Carpeta donde están los PDFs.
    :param output_folder: Carpeta donde se guardarán los .txt.
    """
    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No se encontraron archivos PDF en la carpeta.")
        return

    for pdf_file in pdf_files:
        print(f"Procesando: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        if text.strip():
            txt_path = output / f"{pdf_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Texto extraído y guardado en: {txt_path}")
        else:
            print(f"⚠️ No se pudo extraer texto de: {pdf_file.name}")

if __name__ == "__main__":
    # Carpeta donde tienes tus PDFs
    input_folder = "data/REDES"
    # Carpeta donde se guardarán los TXT extraídos
    output_folder = "data/REDES"

    process_pdfs_in_folder(input_folder, output_folder)