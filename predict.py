import argparse
import helper_functions
import json

def parse_args():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store", type = str)
    parser.add_argument('filepath', default='/home/workspace/ImageClassifier/flowers/test/18/image_04277.jpg', nargs='*', action="store", type = str) 
    parser.add_argument('--top_k', dest='top_k', default='5', action='store')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_data, training_loader, testing_loader, validation_loader = helper_functions.load_the_data()


    model = helper_functions.load_checkpoint(args.checkpoint)


    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)


    ps, classes = helper_functions.predict(args.filepath, model, int(args.top_k))

    for i in range(int(args.top_k)):
        print("\nProbability - {} - Class - {}\n".format(ps[i], classes[i]))
        
if __name__ == "__main__":
    main()
