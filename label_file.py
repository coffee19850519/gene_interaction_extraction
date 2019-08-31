import base64
import json
import os.path
import cfg


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.text_num = 0
        self.arrow_num = 0
        self.inhibit_num = 0
        self.gene_num = 0

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
            'flags',  # image level flags
            'imageHeight',
            'imageWidth',
        ]
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if data['imageData'] is not None:
                image_data = base64.b64decode(data['imageData'])
            else:
                # relative path from label file to relative path from cwd
                # imagePath = os.path.join(cfg.data_dir, cfg.origin_image_dir_name,
                #                         data['imagePath'])
                # parent_path = os.path.dirname(filename)
                # imagePath = os.path.join(parent_path, data['imagePath'])
                # with open(imagePath, 'r') as f:
                #    image_data = f.read()
                image_data = None

            flags = data.get('flags')
            imagePath = data['imagePath']
            lineColor = data['lineColor']
            fillColor = data['fillColor']

            shapes = []
            for s in data['shapes']:
                shapes.append(s)
                category = s['label'].split(':')[0]
                if category == 'inhibit' and s['label'].find('*\t*\t*\t*\t*') == -1:
                    self.inhibit_num = self.inhibit_num + 1
                elif category == 'activate' and s['label'].find('*\t*\t*\t*\t*') == -1:
                    self.arrow_num = self.arrow_num + 1
                else:
                    if s['label'].find('*\t*\t*\t*\t*') == -1:
                        self.text_num = self.text_num + 1

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
        self.imageData = image_data
        self.lineColor = lineColor
        self.fillColor = fillColor
        self.filename = filename
        self.otherData = otherData

    def save(
            self,
            filename,
            shapes,
            imagePath,
            imageHeight,
            imageWidth,
            lineColor=None,
            fillColor=None,
            otherData=None,
            flags=None,
    ):
        if otherData is None:
            otherData = {}

        if flags is None:
            flags = {}

        data = dict(
            version=None,
            flags=flags,
            shapes=shapes,
            lineColor=[0, 255, 0, 128],
            fillColor=[255, 0, 0, 128],
            imagePath=imagePath,
            imageData=None,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            data[key] = value
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    def get_all_genes(self):
        gene_names = []
        for shape in self.shapes:
            if shape['label'].find('gene:') == 0:
                _, _, gene_name = str(shape['label']).partition('gene:')
                gene_names.append(gene_name)
        return gene_names


    def get_all_relations(self):
        relations = []
        for shape in self.shapes:
            if shape['label'].find('|') >= 0 and shape['label'].find('*\t*\t*\t*\t*') == -1:
                relation = str(shape['label'])
                relations.append(relation)
        return relations

    def get_all_text(self):
        text_list = []
        for shape in self.shapes:
            if shape['label'].find('activate:') == -1 and \
                    shape['label'].find('inhibit:') == -1:
                try:
                    _, text = str(shape['label']).split(':', 1)
                except:
                    pass
                if text is not None:
                    text_list.append(text.upper())
        return text_list

    def generate_category(self, shape):
        category = ''
        # if shape['label'].find('arrow') != -1 or shape['label'].find(
        #     '<activate>') != -1 or shape['label'] == 'action:':
        if shape['label'].split(':')[0] == 'activate':
            # arrow
            category = 'arrow'
        elif shape['label'].split(':')[0] == 'inhibit':
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
        # 2019/3/6 WWW get_all_boxes_for_category

    def get_all_shapes_for_category(self, category_name):
        text_boxes = []
        for shape in self.shapes:
            if self.generate_category(shape) == category_name:
                text_boxes.append(shape)
        return text_boxes

    @staticmethod

    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix


if __name__ == '__main__':
    ground_truth_folder = r'C:\Users\LSC-110\Desktop\ground_truth'
    all_genes_in_gt = []
    for json_file in os.listdir(ground_truth_folder):
        if os.path.splitext(json_file)[-1] == '.json':
            json_data = LabelFile(os.path.join(ground_truth_folder, json_file))
            all_genes_in_gt.extend(json_data.get_all_text())
            del json_data
    with open(os.path.join(ground_truth_folder, 'text_list.txt'), 'w') as gene_fp:
        gene_fp.write('\n'.join(all_genes_in_gt))

    # with open(os.path.join(ground_truth_folder, 'gene_list.txt'),'w') as
    # gene_fp:

# end of file
