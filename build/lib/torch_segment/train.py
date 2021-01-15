from .engine import Trainer
from .dataset import get_data_loaders, get_dataset, get_images
from .utils.helpers import visualize_from_dataloader, visualize_from_path
from .utils.helpers import set_colors_list, set_all_classes

import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume-training', dest='resume_training',
                    required=True, help='whether to resume training or not',
                    choices=['yes', 'no'])
parser.add_argument('-p', '--model-path', dest='model_path',
                    help='path to trained model for resuming training')
args = vars(parser.parse_args())

def print_learning_params(learning_params):
    print(learning_params)

def set_model(learning_params):
    model = learning_params['MODEL']
    print(model)
    model.to(learning_params['DEVICE'])
    return model

def init_engine(learning_params, classes_to_train, label_colors_list, all_classes,
                user_train_image_transform=None, user_train_mask_transform=None, 
                user_valid_image_transform=None, user_valid_mask_transform=None,
                show_image=False):

    # set the colors list for helpers.py first
    set_colors_list(label_colors_list)

    # set all classes for helpers.py
    set_all_classes(all_classes)

    print(f"LEARNING PARAMS: {learning_params}")
    # get the data paths
    train_images, train_segs, valid_images, valid_segs = get_images(learning_params['ROOT_PATH'])

    # get datasets from `CamVidDataset()` class
    if (user_train_image_transform != None or 
        user_train_mask_transform != None or
        user_valid_image_transform != None or
        user_valid_mask_transform != None):
        train_dataset, valid_dataset = get_dataset(train_images, train_segs, label_colors_list,
                                                valid_images, valid_segs, classes_to_train,
                                                user_train_image_transform, user_train_mask_transform, 
                                                user_valid_image_transform, user_valid_mask_transform)
    else:
        train_dataset, valid_dataset = get_dataset(train_images, train_segs, label_colors_list,
                                                valid_images, valid_segs, classes_to_train)

    # get data loaders
    train_data_loader, valid_data_loader = get_data_loaders(train_dataset, valid_dataset, 
                                                              learning_params['BATCH_SIZE'])

    if show_image:
        visualize_from_path(train_images, train_segs)

    if show_image:
        visualize_from_dataloader(train_data_loader)


    print(f"\nSAVING CHECKPOINT EVERY {learning_params['SAVE_EVERY']} EPOCHS\n")
    print(f"LOGGING EVERY {learning_params['LOG_EVERY']} EPOCHS\n")
    # initialie `Trainer` if resuming training
    if args['resume_training'] == 'yes':
        if args['model_path'] == None:
            sys.exit('\nPLEASE PROVIDE A MODEL TO RESUME TRAINING FROM!')
        trainer = Trainer( 
        set_model(learning_params), 
        train_data_loader, 
        train_dataset,
        valid_data_loader,
        valid_dataset,
        classes_to_train,
        learning_params['EPOCHS'],
        learning_params['DEVICE'], 
        learning_params['LR'],
        learning_params['VAL_SEG_DIR'],
        learning_params['CHECKPOINTS_DIR'],
        args['resume_training'],
        model_path=args['model_path'],
    )

    # initialie `Trainer` if training from beginning
    else:
        trainer = Trainer( 
            set_model(learning_params), 
            train_data_loader, 
            train_dataset,
            valid_data_loader,
            valid_dataset,
            classes_to_train,
            learning_params['EPOCHS'],
            learning_params['DEVICE'], 
            learning_params['LR'],
            learning_params['VAL_SEG_DIR'],
            learning_params['CHECKPOINTS_DIR'],
            args['resume_training']
        )

    trained_epochs = trainer.get_num_epochs()
    epochs_to_train = learning_params['EPOCHS'] - trained_epochs

    train_loss , train_mIoU, train_pix_acc = [], [], []
    valid_loss , valid_mIoU, valid_pix_acc = [], [], []
    for epoch in range(epochs_to_train):
        epoch = epoch+1+trained_epochs
        print(f"Epoch {epoch} of {learning_params['EPOCHS']}")
        train_epoch_loss, train_epoch_mIoU, train_epoch_pixacc = trainer.fit()
        valid_epoch_loss, valid_epoch_mIoU, valid_epoch_pixacc = trainer.validate(epoch)
        train_loss.append(train_epoch_loss)
        train_mIoU.append(train_epoch_mIoU)
        train_pix_acc.append(train_epoch_pixacc)
        valid_loss.append(valid_epoch_loss)
        valid_mIoU.append(valid_epoch_mIoU)
        valid_pix_acc.append(valid_epoch_pixacc)

        if epoch % learning_params['LOG_EVERY'] == 0: 
            print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch mIoU: {train_epoch_mIoU:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
            print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch mIoU: {valid_epoch_mIoU:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")


        # save model every 5 epochs
        if epoch % learning_params['SAVE_EVERY'] == 0:
            print('SAVING MODEL')
            trainer.save_model(epoch)
            print('SAVING COMPLETE')

    print('TRAINING COMPLETE')