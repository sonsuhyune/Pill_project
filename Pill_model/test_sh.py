import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DATA_DIR', type=str, default="temp.jpg")

    args = parser.parse_args()
    return args

def main(args):
    from_user_img = args.DATA_DIR
    print("ddfsdfsdfas")
    print(from_user_img)

if __name__ == '__main__':
    args = parse_args()
    main(args)