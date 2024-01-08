import argparse
import methods

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    parser.add_argument('data_directory', type=str, help='Directory of the data')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints', default='./')
    parser.add_argument('--arch', type=str, help='Architecture [vgg16, ...]', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Hidden units for classifier', default=512)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=3)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Load and preprocess the data
    dataloaders = methods.load_data(args.data_directory)

    # Initialize the model
    print("Initialize the model")
    model, criterion, optimizer = methods.initialize_model(arch=args.arch, hidden_units=args.hidden_units,
                                                           learning_rate=args.learning_rate,
                                                           class_to_idx=dataloaders['train'].dataset.class_to_idx)

    # Train the model
    print("Train the model")
    methods.train_model(model, criterion, optimizer, dataloaders, epochs=args.epochs, gpu=args.gpu)

    # Save the model checkpoint
    print("Save the model checkpoint")
    methods.save_checkpoint(model, args.save_dir, arch=args.arch, hidden_units=args.hidden_units)

if __name__ == '__main__':
    main()
