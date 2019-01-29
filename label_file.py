import base64
import json
import os.path


class LabelFileError(Exception):
    pass

class LabelFile(object):

    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        keys = [
            'imageData',
            'imagePath',
            'lineColor',
            'fillColor',
            'shapes',  # polygonal annotations
            'flags',   # image level flags
            'imageHeight',
            'imageWidth',
        ]
        try:
            with open(filename, 'rb') as f:
                data = json.load(f)
            if data['imageData'] is not None:
                imageData = base64.b64decode(data['imageData'])
            else:
                # relative path from label file to relative path from cwd
                imagePath = os.path.join(os.path.dirname(filename),
                                         data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
            flags = data.get('flags')
            imagePath = data['imagePath']
            lineColor = data['lineColor']
            fillColor = data['fillColor']

            shapes = []
            for s in data['shapes']:
                shapes.append(s)

        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.lineColor = lineColor
        self.fillColor = fillColor
        self.filename = filename
        self.otherData = otherData

    def get_all_genes(self):
        gene_names = []
        for shape in self.shapes:
            _,_,gene_name = str(shape['labels']).partition('gene:')
            gene_names.append(gene_name)
        return gene_names

    def generate_category(self, shape):
        category = ''
        if shape['label'].find('arrow') != -1 or shape['label'].find(
            'activate') != -1:
            # arrow
            category = 'arrow'
        elif shape['label'].find('action:inhibit') != -1:
            category = 'nock'
        else:
            category = 'text'
        return category

    def get_all_boxes_for_category(self, category_name):
        text_boxes = []
        for shape in self.shapes:
            if self.generate_category(shape) == category_name:
                text_boxes.append(shape['points'])
        return text_boxes


    @staticmethod
    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix
