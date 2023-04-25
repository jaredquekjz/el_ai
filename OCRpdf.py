import re
import sys
from pdf2image import convert_from_path
import pytesseract
from pagelabels import PageLabels
from pdfrw import PdfReader as PdfReader_pdfrw

input_file = "/Users/e/Python/ChatGPT Programs/Storage/Texts/Boom.pdf"
output_file = "/Users/e/Python/ChatGPT Programs/Storage/Texts/Boom.txt"


def preprocess_text(text):
    text = re.sub(r'\d', '', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text


def split_into_sentences(text):
    sentences = re.split(r'\.(\s|\n)', text)
    sentences = [sentence.strip()
                 for sentence in sentences if sentence.strip()]
    sentences = [sentence + '.' for sentence in sentences]
    return sentences


def main():
    # Convert PDF to images
    images = convert_from_path(input_file)
    reader_pdfrw = PdfReader_pdfrw(input_file)
    page_labels = PageLabels.from_pdf(reader_pdfrw)

    print(f"Processing {len(images)} pages...")

    all_chunks = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        text = preprocess_text(text)
        sentences = split_into_sentences(text)

        if i < len(page_labels):
            label = page_labels[i].label
        else:
            label = i + 1

        for sentence in sentences:
            # Replace the newline character in the sentence with a space
            sentence = sentence.replace('\n', ' ')
            all_chunks.append(
                f"From 'Boom', Page {label}: {sentence}\n")

        print(f"Processed page {i+1}/{len(images)}")

    with open(output_file, "w") as txt_file:
        txt_file.writelines(all_chunks)

    print(f"Text file saved as '{output_file}'")


if __name__ == "__main__":
    main()
