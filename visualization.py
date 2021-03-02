from pathlib import Path
import ast
import os
import threading
import webbrowser

def parse():
    """
    Load args from file. Tries to convert them to best type.
    """
    try:
        # Process input
        hyperparams_path = Path('./')
        if not hyperparams_path.exists():
            raise OSError('Setup dir not found')
        if hyperparams_path.is_dir():
            hyperparams = os.path.join(hyperparams_path, 'visualizationSetup.txt')
        # Prepare output
        output = {}
        # Read file
        with open(hyperparams) as file:
            # Read lines
            for l in file:
                if l.startswith('#'):
                    continue
                # Remove new line
                l = l.strip()
                # Separate name from value
                toks = l.split(':')
                name = toks[0]
                value = ':'.join(toks[1:]).strip()
                # Parse value
                try:
                    value = ast.literal_eval(value)
                except:
                    pass
                # Add to output
                output[name] = value

            # Verify setup integrity
            if not all(key in output.keys() for key in ['logs_dir',
                                            'tensorboard_port']):
                raise AttributeError("Params consistency broken!")
    except (FileNotFoundError, AttributeError, Exception):
        print("Restoring original params value in the setup file... please try to reconfigure setup.")
        f = open(os.path.join(hyperparams_path, 'visualizationSetup.txt'), 'w')
        f.write("###################################################################################\n\
# VISUALIZATION SETUP: please don't delete or modify attributes name (before ':') #\n\
###################################################################################\n\
\n\
# Logs options\n\
logs_dir: './logs'\n\
tensorboard_port: 6006")
        f.close()
        raise AttributeError("Exit")

    # Return
    return output
    
if __name__ == '__main__':
    # Get params
    args = parse()

    # Retrieve absolute path of logs folder
    logs = os.path.abspath(args['logs_dir'])

    # Start TensorBoard Daemon to visualize data
    print("Starting TensorBoard... please, wait a bit to loading all results.")
    tensorboard_port = args['tensorboard_port']
    t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(logs) + ' --port=' + str(tensorboard_port) + ' --bind_all'))
    t.start()
    webbrowser.open('http://localhost:' + str(tensorboard_port) + '/', new=1)