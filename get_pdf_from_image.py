import glob

main_pdf_directory = r'/Users/orion/Desktop/use_case1/pdfs'
def pdf_from_image_name(image_name):
    splitname = image_name.split('_');
    subid = splitname[1]
    firstword = splitname[2] + ' '
    pdfs = []
    for folder in glob.glob(main_pdf_directory + '/*'):
        found = False
        for pdf in glob.glob(folder + '/{}*_{}_.pdf'.format(firstword, subid)):
            pdfs = [pdf]
            found = True
        if found:
            break
        if not found:
            for pdf in glob.glob(folder + '/{}*.pdf'.format(firstword)):
                pdfs.append(pdf)
    return pdfs


if __name__ == "__main__":
    image_directory = r'/Users/orion/Desktop/use_case1/images'
    for image in glob.glob(image_directory + r'\*.png'):
        imagename = image.split('\\')[-1]
        possible_pdfs = pdf_from_image_name(imagename)
        if len(possible_pdfs) > 1:
            print(imagename)
            for possible_pdf in possible_pdfs:
                print(possible_pdf)
