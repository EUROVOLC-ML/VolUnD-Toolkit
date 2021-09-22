import os
import threading
import webbrowser
from utils.parser import visualization_parse


if __name__ == '__main__':
    # Get params
    args = visualization_parse()

    # Retrieve absolute path of logs folder
    logs = os.path.abspath(args['logs_dir'])

    # Start TensorBoard Daemon to visualize data
    print("Starting TensorBoard... please, wait a bit to load all results.")
    tensorboard_port = args['tensorboard_port']
    t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(logs) + ' --port=' + str(tensorboard_port) + ' --bind_all'))
    t.start()
    webbrowser.open('http://localhost:' + str(tensorboard_port) + '/', new=1)
