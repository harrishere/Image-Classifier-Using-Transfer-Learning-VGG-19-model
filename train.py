  
import argparse
import helper_functions

def parse_args():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--data_dir', action='store', default="./flowers/")
    parser.add_argument('--epochs', dest='epochs', default='1')
    parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')
    parser.add_argument('--hidden_units', dest='hidden_units', default='2048')
    parser.add_argument('--dropout', dest = "dropout", action = "store", default = '0.5')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', dest='arch', default='vgg19', choices=['vgg16', 'vgg19'])
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_data, trainloader, validloader, testloader = helper_functions.load_the_data(args.data_dir)

    model, criterion, optimizer = helper_functions.define_network(args.arch, int(args.hidden_units), float(args.dropout), float(args.learning_rate), args.gpu)


    helper_functions.train_network(model, criterion, optimizer, trainloader, validloader, int(args.epochs), args.gpu)


    helper_functions.save_checkpoint(args.save_dir, model, optimizer, train_data, args)


    print("Congrats, The Model is trained now") 
    
    
if __name__ == "__main__":
    main()