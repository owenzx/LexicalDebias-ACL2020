import os
import sys

TESTING_DATASETS = 'cwb_enh_test'

BIAS_TEST = True

TEST_MODEL = -3 #best checkpoint

NEW_GPU = 3





def get_test_command_from_train(command):

    # Delete output redirection
    if '>>' in command:
        command = command.split('>>')[0]
    elif '>' in command:
        command = command.split('>')[0]

    #Correct train/test flags
    if "--train=True" in command:
        command.replace('--train=True', '--train=False')
    else:
        command += ' --train=False'

    #Correct load_dict
    if "--load_dict=True" in command:
        command.replace('--load_dict=True', '--load_dict=False')
    else:
        command += " --load_dict=True"

    #Bias test option
    if BIAS_TEST:
        command += " --bias_test=True --major_class=2"

    #Model selection
    if TEST_MODEL is not None:
        command += " --test_iter=%d"%TEST_MODEL
    
    #delete part load path
    if "--part_load_path" in command:
        old_test_idx_l = command.find('--part_load_path=')
        old_test_idx_r = command[old_test_idx_l:].find(' --') + old_test_idx_l
        command = command[:old_test_idx_l] + command[old_test_idx_r:]

    #Correct testing datasets
    if TESTING_DATASETS is not None:
        old_test_idx_l = command.find('--test_datasets=')
        old_test_idx_r = command[old_test_idx_l:].find(' --') + old_test_idx_l
        command = command[:old_test_idx_l] + command[old_test_idx_r:]
        command += (' --test_datasets=' + TESTING_DATASETS)

    #Set GPU
    if NEW_GPU is not None:
        old_test_idx_l = command.find('--gpu_id=')
        old_test_idx_r = command[old_test_idx_l:].find(' --') + old_test_idx_l
        command = command[:old_test_idx_l] + command[old_test_idx_r:]
        command += (' --gpu_id=' + str(NEW_GPU))

    return command


def main():
    train_file = sys.argv[1]
    with open(train_file, 'r') as fr:
        train_command  = fr.read() 

    # handle multiple runs
    commands = train_command.split('python')
    for c in commands:
        if len(c)==0:
            continue
        if c[-1]=='\n':
            c = c[:-1]
        c = 'python' + c

        test_command = get_test_command_from_train(c)
        
        print(test_command)

        os.system(test_command)


if __name__ == "__main__":
    main()
