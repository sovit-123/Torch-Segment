from torch_segment.utils.helpers import ALL_CLASSES


def set_learning_params(model=None, epochs=75, save_every=5, log_every=5, batch_size=16, 
                        device='cuda', lr=0.0001, root_path='camvid',
                        val_seg_dir='./', checkpoints_dir='./'):
        learning_params = dict()

        learning_params = {
                'MODEL': model,
                'EPOCHS': epochs,
                'SAVE_EVERY': save_every, # after how many epochs to save a checkpoint
                'LOG_EVERY': log_every, # log training and validation metrics every `LOG_EVERY` epochs
                'BATCH_SIZE': batch_size,
                'DEVICE': device,
                'LR': lr,
                'ROOT_PATH': root_path,
                'VAL_SEG_DIR': val_seg_dir, 
                'CHECKPOINTS_DIR': checkpoints_dir 
        }

        return learning_params

def set_classes_to_train(classes_to_train=None):
        # the classes that we want to train
        if classes_to_train == None:
                CLASSES_TO_TRAIN = [
                        'animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
                        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
                        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
                        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk',
                        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
                        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
                        'wall'
                        ]
                return CLASSES_TO_TRAIN
        else:
                return classes_to_train

def set_all_classes(all_classes=None):
        if all_classes == None:
                ALL_CLASSES = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
                'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
                'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving', 
                'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk', 
                'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
                'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
                'wall']
                return ALL_CLASSES
        else:
                return all_classes

def set_debug(debug=True):
        # DEBUG for visualizations
        DEBUG = debug
        return DEBUG

def set_label_colors_list(label_colors_list=None):
        if label_colors_list == None:
                LABEL_COLORS_LIST = [
                        (64, 128, 64), # animal
                        (192, 0, 128), # archway
                        (0, 128, 192), # bicyclist
                ]
                return LABEL_COLORS_LIST

        else:
                return label_colors_list