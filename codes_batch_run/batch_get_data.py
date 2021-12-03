import os

for PREV_TEXT_ORDER in [0, 1]:
    for CUR_PREV_ORDER in [0, 1]:
        for ADD_SCENE_TEXT in [0, 1]:
            for ADD_CHARACTER_TEXT in [0, 1]:
                for MAX_SEQ_LEN in [380, 400, 420]:
                    os.system(f'python get_data_v1.py {PREV_TEXT_ORDER} {CUR_PREV_ORDER} {ADD_SCENE_TEXT} {ADD_CHARACTER_TEXT} {MAX_SEQ_LEN}' )
for PREV_TEXT_ORDER in [0, 1]:
    for CUR_PREV_ORDER in [0, 1]:
        for ADD_SCENE_TEXT in [0, 1]:
            for ADD_CHARACTER_TEXT in [0, 1]:
                for MAX_SEQ_LEN in [380, 400, 420]:
                    os.system(f'python get_data_v2.py {PREV_TEXT_ORDER} {CUR_PREV_ORDER} {ADD_SCENE_TEXT} {ADD_CHARACTER_TEXT} {MAX_SEQ_LEN}' )
