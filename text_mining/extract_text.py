import textract
import glob
import os
import sys


def convert_file_to_text(source):
    paperInText = textract.process(source).decode()
    return paperInText

def write_pdf_as_text_file(source, destination_folder):
    convert_file_to_text(source)
    pdfName = source[:-4]
    filename = destination_folder + os.path.basename(pdfName) + '.txt'
    with open(filename, 'w') as f:
        f.write(paperInText)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        if len(sys.argv) > 2:
            destination_folder = sys.argv[2]
        else:
            destination_folder = ''
        write_pdf_as_text_file(source, destination_folder)
