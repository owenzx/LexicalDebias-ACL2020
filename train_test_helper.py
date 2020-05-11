import tensorflow as tf
from utils import get_value_summary, concat_all_batches, get_real_acc, backup_best_model, get_new_numbers
from constants import *
import numpy as np
import sys
from tqdm import tqdm

def test_supervised(params, model, saver, sess, exp_string, data_generator, resume_epoch=0):
    save_result = params["save_result"]

    results_dict = {}

    for dataset_name in data_generator.test_dataset_names:
        test_loss, test_acc, results_dict[dataset_name] = test_on_dataset(params, model, sess, data_generator, dataset_name)
        print('Test results on %s: '%dataset_name + '  loss: ' + str(test_loss) + ',  acc: ' + str(test_acc))
        sys.stdout.flush()



def biased_test(params, model, saver, sess, exp_string, data_generator, resume_itr, major_class):
    for dataset_name in data_generator.test_dataset_names:
        if 'nwb' in dataset_name:
            major_class = 2 
        elif 'wob' in dataset_name:
            major_class = 0
        else:
            major_class = major_class

        print('Test results on %s: '%dataset_name)

        test_on_dataset_biased(params, model, sess, data_generator, dataset_name, major_class)

        sys.stdout.flush()



def train_supervised(params,model, saver, sess, exp_string, data_generator, resume_epoch=0):
    SUMMARY_INTERVAL = 1000
    SAVE_INTERVAL = 1
    PRINT_INTERVAL = 10000
    TEST_PRINT_INTERVAL = 1*PRINT_INTERVAL

    pbar = tqdm(total = TEST_PRINT_INTERVAL)

    if params["log"]:
        train_writer = tf.summary.FileWriter(params["logdir"] + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')

    nlilosses, nliacces = [], []

    best_val_acc = 0.0
    new_best_model = False

    for itr, batch in enumerate(data_generator.get_train_batch()):        
        if itr > params["pretrain_itrs"]:
            pbar.close()
            break
        pbar.update(1)

        feed_dict = {data_generator.prem_ph: batch['prem'],
                    data_generator.hypo_ph: batch['hypo'],
                    model.keep_rate_ph: 1-params["dropout_rate"],
                    model.stop_grad_ph: 0}

        if batch['task'] == 'nli':
            #Set placeholders
            feed_dict[data_generator.nli_y_ph] = batch['y']

            input_tensors = [model.sum_train_op]

            input_tensors.extend([model.nli_loss, model.nli_acc])


            result = sess.run(input_tensors, feed_dict=feed_dict)

            nlilosses.append(result[1])
            nliacces.append(result[2])                

            if params["log"] and itr % SUMMARY_INTERVAL ==0:
                get_value_summary(tag = 'train/loss', value = result[1], writer = train_writer, itr=itr)
                get_value_summary(tag = "train/acc", value = result[2], writer = train_writer, itr=itr)
                train_writer.flush()


        if itr % PRINT_INTERVAL == 0:
            print_str = 'Train Itr ' + str(itr)
            if len(nlilosses)>0:
                print_str += ' NLI Loss: ' + str(np.mean(nlilosses)) + ',  NLI Acc: ' + str(np.mean(nliacces))
            print(print_str)
            sys.stdout.flush()
            nlilosses, nliacces = [], []

        feed_dict = {}

        if itr % TEST_PRINT_INTERVAL == 0 and itr!=0:
            pbar.close()
            #Test on in-domain and out-of-domain nli datasets
            for val_idx, dataset_name in enumerate(data_generator.test_dataset_names):
                test_loss, test_acc, _ = test_on_dataset(params, model, sess, data_generator, dataset_name)
                            
                if params["log"]:
                    get_value_summary(tag = "dev_loss/%s"%dataset_name, value=test_loss, writer=train_writer, itr = itr)
                    get_value_summary(tag = "dev_acc/%s"%dataset_name, value=test_acc, writer=train_writer, itr=itr)
                    train_writer.flush()
                print('Validation results on %s: '%dataset_name + '  loss: ' + str(test_loss) + ',  acc: ' + str(test_acc))
                sys.stdout.flush()

                if val_idx == params["main_val_idx"]:
                    if (test_acc > best_val_acc):
                        new_best_model = True
                        best_val_acc = test_acc

                
            saver.save(sess, params["logdir"] + '/' + exp_string +  '/pretrain_model' + str(itr))
            if new_best_model:
                best_path = params["logdir"] + '/' + exp_string +  '/pretrain_model_BEST'
                saver.save(sess, best_path)
                backup_best_model(params["logdir"], best_path)

            new_best_model = False
            pbar = tqdm(total=TEST_PRINT_INTERVAL)



def test_on_dataset_biased(params, model,  sess, data_generator, dataset_name, major_class=2):
    SUMMARY_INTERVAL = 1000
    SAVE_INTERVAL = 1
    PRINT_INTERVAL = 10000
    TEST_PRINT_INTERVAL = 1*PRINT_INTERVAL
    
    is_twoway=False

    nlilosses, nliacces = [], []
    preds, labels = [], []

    for itr, batch in enumerate(data_generator.get_test_batch(dataset_name)):

        feed_dict = {data_generator.prem_ph: batch['prem'],
                     data_generator.hypo_ph: batch['hypo'],
                     data_generator.nli_y_ph: batch['y'],
                     model.keep_rate_ph: 1,
                     model.stop_grad_ph: 0}
        

        labels.extend(batch['y'])
        
        input_tensors = [model.nli_loss, model.nli_acc, model.output]
        result = sess.run(input_tensors, feed_dict = feed_dict)

        preds.extend(result[2])

        nlilosses.append(result[0])
        nliacces.append(get_real_acc(result[1], result[2], batch['y'], is_twoway))

    
    get_new_numbers(preds, labels, major_class)

    print_str = 'Test: '
    if len(nlilosses)>0:
        print_str += ' NLI Loss: ' + str(np.mean(nlilosses)) + ',  NLI Acc: ' + str(np.mean(nliacces))
    print(print_str)
    sys.stdout.flush()



def test_hex(params, model, saver, sess, exp_string, data_generator, resume_itr):
    SUMMARY_INTERVAL = 1000
    SAVE_INTERVAL = 1
    PRINT_INTERVAL = 10000
    TEST_PRINT_INTERVAL = 1*PRINT_INTERVAL

    pbar = tqdm(total=TEST_PRINT_INTERVAL)

    if params['log']:
        train_writer = tf.summary. FileWriter(params["logdir"]+ '/' + exp_string, sess.graph)
    print('Done initiaizing, starting training.')
    nlilosses, nliacces = [], []


    for itr, batch in enumerate(data_generator.get_train_batch()):
        pbar.update(1)

        feed_dict = {data_generator.prem_ph: batch['prem'],
                     data_generator.hypo_ph: batch['hypo'],
                     data_generator.nli_y_ph: batch['y'],
                     model.keep_rate_ph: 1,
                     model.stop_grad_ph: 0}
        
        input_tensors = [model.nli_loss_test, model.nli_acc_test]
        result = sess.run(input_tensors, feed_dict = feed_dict)

        nlilosses.append(result[1])
        nliacces.append(result[2])                

        if params["log"] and itr % SUMMARY_INTERVAL ==0:
            get_value_summary(tag = 'test/loss', value = result[1], writer = train_writer, itr=itr)
            get_value_summary(tag = "test/acc", value = result[2], writer = train_writer, itr=itr)
            train_writer.flush()

        print_str = 'Test: '
        if len(nlilosses)>0:
            print_str += ' NLI Loss: ' + str(np.mean(nlilosses)) + ',  NLI Acc: ' + str(np.mean(nliacces))
        print(print_str)
        sys.stdout.flush()


def train_hex(params, model, saver, sess, exp_string,data_generator, resume_itr):
    SUMMARY_INTERVAL = 1000
    SAVE_INTERVAL = 1
    PRINT_INTERVAL = 10000
    TEST_PRINT_INTERVAL = 1*PRINT_INTERVAL

    pbar = tqdm(total=TEST_PRINT_INTERVAL)

    if params['log']:
        train_writer = tf.summary. FileWriter(params["logdir"]+ '/' + exp_string, sess.graph)
    print('Done initiaizing, starting training.')
    nlilosses_g, nlilosses_l, nlilosses_p, nliacces_g, nliacces_l, nliacces_p = [], [], [], [], [], []

    best_val_acc = 0.0
    new_best_model = False

    for itr, batch in enumerate(data_generator.get_train_batch()):
        if itr > params["pretrain_itrs"]:
            pbar.close()
            break
        pbar.update(1)

        feed_dict = {data_generator.prem_ph: batch['prem'],
                     data_generator.hypo_ph: batch['hypo'],
                     data_generator.nli_y_ph: batch['y'],
                     model.keep_rate_ph: 1 - params["dropout_rate"],
                     model.stop_grad_ph: 0}
        
        if itr < params['freeze_sf_itr']:
            input_tensors = [model.train_op, model.nli_loss_train_l, model.nli_loss_train_g, model.nli_loss_test, model.nli_acc_train_l, model.nli_acc_train_g, model.nli_acc_test]
        else:
            input_tensors = [model.train_op_2, model.nli_loss_train_l, model.nli_loss_train_g, model.nli_loss_test, model.nli_acc_train_l, model.nli_acc_train_g, model.nli_acc_test]

        result = sess.run(input_tensors, feed_dict = feed_dict)

        nlilosses_g.append(result[1])
        nlilosses_l.append(result[2])
        nlilosses_p.append(result[3])
        nliacces_g.append(result[4])                
        nliacces_l.append(result[5])                
        nliacces_p.append(result[6])

        if params["log"] and itr % SUMMARY_INTERVAL ==0:
            get_value_summary(tag = 'train/loss_g', value = result[1], writer = train_writer, itr=itr)
            get_value_summary(tag = 'train/loss_l', value = result[2], writer = train_writer, itr=itr)
            get_value_summary(tag = 'train/loss_p', value = result[3], writer = train_writer, itr=itr)
            get_value_summary(tag = "train/acc_g", value = result[4], writer = train_writer, itr=itr)
            get_value_summary(tag = "train/acc_l", value = result[5], writer = train_writer, itr=itr)
            get_value_summary(tag = "train/acc_p", value = result[6], writer = train_writer, itr=itr)
            train_writer.flush()

        if itr % PRINT_INTERVAL == 0:
            print_str = 'Train Itr ' + str(itr)
            if len(nlilosses_g)>0:
                print_str += ' NLI Loss g: ' + str(np.mean(nlilosses_g)) + ',  NLI Acc g: ' + str(np.mean(nliacces_g))
                print_str += ' NLI Loss l: ' + str(np.mean(nlilosses_l)) + ',  NLI Acc l: ' + str(np.mean(nliacces_l))
            print(print_str)
            sys.stdout.flush()
            nlilosses_g, nlilosses_l, nlilosses_p, nliacces_g, nliacces_l, nliacces_p = [], [], [], [], [], []
        if itr% TEST_PRINT_INTERVAL == 0 and itr!=0:
            pbar.close()
            best_val_acc = validation_nli_multiple_datasets(params, model, saver, sess, exp_string, data_generator, train_writer, itr, best_val_acc)
            pbar = tqdm(total=TEST_PRINT_INTERVAL)



def validation_nli_multiple_datasets(params, model, saver, sess, exp_string, data_generator, writer, itr, best_val_acc):
    #Test on in-domain and out-of-domain nli datasets
    new_best_model = False
    for val_idx, dataset_name in enumerate(data_generator.test_dataset_names):
        test_loss, test_acc, _ = test_on_dataset(params, model, sess, data_generator, dataset_name)
                    
        if params["log"]:
            get_value_summary(tag = "dev_loss/%s"%dataset_name, value=test_loss, writer=writer, itr = itr)
            get_value_summary(tag = "dev_acc/%s"%dataset_name, value=test_acc, writer=writer, itr=itr)
            writer.flush()
        print('Validation results on %s: '%dataset_name + '  loss: ' + str(test_loss) + ',  acc: ' + str(test_acc))
        sys.stdout.flush()

        if val_idx == params["main_val_idx"]:
            if (test_acc > best_val_acc):
                new_best_model = True
                best_val_acc = test_acc

        
    saver.save(sess, params["logdir"] + '/' + exp_string +  '/pretrain_model' + str(itr))
    if new_best_model:
        best_path = params["logdir"] + '/' + exp_string +  '/pretrain_model_BEST'
        saver.save(sess, best_path)
        backup_best_model(params["logdir"], best_path)

    return best_val_acc




def test_on_dataset(params, model, sess, data_generator, dataset_name, print_total_loss=None):
    if print_total_loss == None:
        total_loss = params['best_loss']
    else:
        total_loss = print_total_loss
    save_result = params["save_result"]

    is_NLI_dataset = True
    is_twoway = False

    test_losses, test_acces = [], []
    feed_dict = {}
    results_dict = {'pred':[], 'label':[], 'inp':[], 'correct':[], 'prem':[], 'hypo':[]}

    if is_NLI_dataset:
        for batch in data_generator.get_test_batch(dataset_name):
            try:
                feed_dict = {data_generator.prem_ph: batch['prem'],
                            data_generator.hypo_ph: batch['hypo'],
                            data_generator.nli_y_ph: batch['y'],
                            model.keep_rate_ph: 1,
                            model.stop_grad_ph: 0}


                if total_loss:
                    input_tensors = [model.nli_label_loss, model.nli_acc, model.output]
                else:
                    input_tensors = [model.nli_loss, model.nli_acc, model.output]
                result = sess.run(input_tensors, feed_dict=feed_dict)
                test_losses.append(result[0])
                test_acces.append(get_real_acc(result[1], result[2], batch['y'], is_twoway))
                if save_result:
                    pred = result[2]
                    true_label = batch['y']
                    results_dict['pred'].append(pred)
                    results_dict['label'].append(true_label)
                    results_dict['correct'].append(pred==true_label)

            except tf.errors.OutOfRangeError:
                break
        if save_result:
            results_dict['inp'] = data_generator.get_test_inps(dataset_name)
            results_dict['prem'] = results_dict['inp']['prem']
            results_dict['hypo'] = results_dict['inp']['hypo']
            results_dict['pred'] = concat_all_batches(results_dict['pred'])
            results_dict['label'] = concat_all_batches(results_dict['label'])
            results_dict['correct'] = concat_all_batches(results_dict['correct'])

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_acces)

    return test_loss, test_acc, results_dict
