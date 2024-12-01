import  json

conf = json.load(open('Configs/template_spec.json'))

DOC_WIDTH = conf['doc_width']
DOC_HEIGHT = conf['doc_height']
GAP_BETWEEN_EVERY_ANSWER_COL = conf['gap_between_every_answer_col']
ANSWER_COL_WIDTH = conf['answer_col_width']
DIS_BETWEEN_BOUNDED_BOXES = conf['dis_between_bounded_boxes']
GAP_BETWEEN_EVERY_ANSWERS_BLOCK = conf['gap_between_every_answers_block']
ANSWERS_COL_X_COORDINATE = conf['answers_col_x_coordinate']
BOUNDED_BOX_MIN_AREA = conf['bounded_box_min_area']
BOUNDED_BOX_MAX_AREA = conf['bounded_box_max_area']