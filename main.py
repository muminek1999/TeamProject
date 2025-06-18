import sys
from train import train_model
from use import classify_url
from test import test_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train <csv_path> | use <url> | test <csv_path>]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train" and len(sys.argv) == 3:
        train_model(sys.argv[2])
    elif command == "use" and len(sys.argv) == 3:
        classify_url(sys.argv[2])
    elif command == "test" and len(sys.argv) == 3:
        test_model(sys.argv[2])
    else:
        print("Invalid arguments. Usage:")
        print(" python main.py train <path_to_csv>")
        print(" python main.py test <path_to_csv>")
        print(" python main.py use <url>")
        sys.exit(1)
