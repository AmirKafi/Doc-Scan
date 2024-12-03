import  json
import os

base_root = os.path.abspath(os.path.dirname(__file__))

conf = json.load(open(os.path.join(base_root, 'template_spec.json')))

DOC_WIDTH = conf['doc_width']
DOC_HEIGHT = conf['doc_height']
ANSWER_COL_WIDTH = conf['answer_col_width']
DIS_BETWEEN_BOUNDED_BOXES = conf['dis_between_bounded_boxes']
GAP_BETWEEN_EVERY_ANSWERS_BLOCK = conf['gap_between_every_answers_block']
ANSWERS_COL_X_COORDINATE = conf['answers_col_x_coordinate']
BOUNDED_BOX_MAX_AREA = conf['bounded_box_max_area']
DIS_TO_FIRST_BOUNDED_BOX = conf['dis_to_first_bounded_box']