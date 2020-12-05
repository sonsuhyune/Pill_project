import os
import time
import re
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils.crnn_utils import CTCLabelConverter, AttnLabelConverter, Averager
from crnn_model import CRNN
from configs import crnn_opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_all_eval(model, criterion, evaluation_loader, converter, crnn_opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = crnn_opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    # log = open(f'./result/{crnn_opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    # log.write(dashed_line + '\n')
    # for eval_data in eval_data_list:
    #     eval_data_path = os.path.join(crnn_opt.eval_data, eval_data)
    #     AlignCollate_evaluation = AlignCollate(imgH=crnn_opt.imgH, imgW=crnn_opt.imgW, keep_ratio_with_pad=crnn_opt.PAD)
    #     eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, crnn_opt=crnn_opt)
    #     evaluation_loader = torch.utils.data.DataLoader(
    #         eval_data, batch_size=evaluation_batch_size,
    #         shuffle=False,
    #         num_workers=int(crnn_opt.workers),
    #         collate_fn=AlignCollate_evaluation, pin_memory=True)
    _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
        model, criterion, evaluation_loader, converter, crnn_opt)
    list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
    total_forward_time += infer_time
    # total_evaluation_data_number += len(eval_data)
    total_evaluation_data_number += 1
    total_correct_number += accuracy_by_best_model * length_of_data
    # log.write(eval_data_log)
    print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
    # log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
    print(dashed_line)
    # log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    # log.write(evaluation_log + '\n')
    # log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, crnn_opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    #log = open(f'result/predict_and_gt.txt', 'a')
    image_tensors = evaluation_loader
    #labels = evaluation_loader[1]
    #log.write("labels: "+str(labels)+"\n")
    #print("labels: ")
    #print(labels)
    batch_size = image_tensors.size(0)
    length_of_data = length_of_data + batch_size
    image = image_tensors.to(device)
    # For max length prediction
    length_for_pred = torch.IntTensor([crnn_opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, crnn_opt.batch_max_length + 1).fill_(0).to(device)

    #text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=crnn_opt.batch_max_length)

    start_time = time.time()
    if 'CTC' in crnn_opt.Prediction:
        preds = model(image, text_for_pred)
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # permute 'preds' to use CTCloss format
        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)

    else:
        preds = model(image, text_for_pred, is_train=False)
        forward_time = time.time() - start_time

        #preds = preds[:, :text_for_loss.shape[1] - 1, :]
        #target = text_for_loss[:, 1:]  # without [GO] Symbol
        # cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        #labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        #postprocess
        #labels = [ label[:label.find('[s]')] for label in labels]
        preds = [ pred[:pred.find('[s]')] for pred in preds_str]

    # infer_time += forward_time
    # valid_loss_avg.add(cost)

    # # calculate accuracy & confidence score
    # preds_prob = F.softmax(preds, dim=2)
    # preds_max_prob, _ = preds_prob.max(dim=2)
    # confidence_score_list = []
    # for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
    #     if 'Attn' in crnn_opt.Prediction:
    #         gt = gt[:gt.find('[s]')]
    #         pred_EOS = pred.find('[s]')
    #         pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
    #         pred_max_prob = pred_max_prob[:pred_EOS]
    #         print("pred: ")
    #         print(pred)
    #     # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
    #     if crnn_opt.sensitive and crnn_opt.data_filtering_off:
    #         pred = pred.lower()
    #         gt = gt.lower()
    #         alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
    #         out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
    #         pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
    #         gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
    #
    #     if pred == gt:
    #         n_correct += 1
    #
    #     '''
    #     (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
    #     "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
    #     if len(gt) == 0:
    #         norm_ED += 1
    #     else:
    #         norm_ED += edit_distance(pred, gt) / len(gt)
    #     '''
    #
    #     # ICDAR2019 Normalized Edit Distance
    #     if len(gt) == 0 or len(pred) == 0:
    #         norm_ED += 0
    #     elif len(gt) > len(pred):
    #         norm_ED += 1 - edit_distance(pred, gt) / len(gt)
    #     else:
    #         norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    #
    #     # calculate confidence score (= multiply of pred_max_prob)
    #     try:
    #         confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    #     except:
    #         confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
    #     confidence_score_list.append(confidence_score)
    #     # print(pred, gt, pred==gt, confidence_score)
    #
    # accuracy = n_correct / float(length_of_data) * 100
    # norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    # return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data
    
    return preds

def recognize_text(batch_text_image):
    """ model configuration """
    if 'CTC' in crnn_opt.Prediction:
        converter = CTCLabelConverter(crnn_opt.character)
    else:
        converter = AttnLabelConverter(crnn_opt.character)
    crnn_opt.num_class = len(converter.character)
    #log = open(f'result/predict_and_gt.txt', 'a')
    if crnn_opt.rgb:
        crnn_opt.input_channel = 3

    if crnn_opt.sensitive:
        crnn_opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    # model = CRNN(crnn_opt)
    # print('model input parameters', crnn_opt.imgH, crnn_opt.imgW, crnn_opt.num_fiducial, crnn_opt.input_channel, crnn_opt.output_channel,
    #       crnn_opt.hidden_size, crnn_opt.num_class, crnn_opt.batch_max_length, crnn_opt.Transformation, crnn_opt.FeatureExtraction,
    #       crnn_opt.SequenceModeling, crnn_opt.Prediction)
    # model = torch.nn.DataParallel(model).to(device)
    #
    # # load model
    # print('loading pretrained model from %s' % crnn_opt.saved_model)
    # model.load_state_dict(torch.load(crnn_opt.saved_model, map_location=device))
    # crnn_opt.exp_name = '_'.join(crnn_opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    # os.makedirs(f'./result/{crnn_opt.exp_name}', exist_ok=True)
    # os.system(f'cp {crnn_opt.saved_model} ./result/{crnn_opt.exp_name}/')

    """ setup loss """
    if 'CTC' in crnn_opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    crnn_net = CRNN(crnn_opt)
    crnn_net = torch.nn.DataParallel(crnn_net, device_ids=[0]).to(device)
    crnn_net.load_state_dict(torch.load('./weights/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth', map_location=device))
    #log.write("==================================================================="+"\n")
    #log.write('Finished loading CRNN model!'+'\n')
    #print('Finished loading CRNN model!')
    crnn_net.eval()

    with torch.no_grad():
        #evaluation_loader = (batch_text_image, labels)
        evaluation_loader = batch_text_image
        if crnn_opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(crnn_net, criterion, evaluation_loader, converter, crnn_opt)
        # log = open(f'./result/{crnn_opt.exp_name}/log_evaluation.txt', 'a')
        # AlignCollate_evaluation = AlignCollate(imgH=crnn_opt.imgH, imgW=crnn_opt.imgW, keep_ratio_with_pad=crnn_opt.PAD)
        # eval_data, eval_data_log = hierarchical_dataset(root=crnn_opt.eval_data, crnn_opt=crnn_opt)
        # evaluation_loader = torch.utils.data.DataLoader(
        #     eval_data, batch_size=crnn_opt.batch_size,
        #     shuffle=False,
        #     num_workers=int(crnn_opt.workers),
        #     collate_fn=AlignCollate_evaluation, pin_memory=True)
        # _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
        #     crnn_net, criterion, evaluation_loader, converter, crnn_opt)
        pred =  validation(crnn_net, criterion, evaluation_loader, converter, crnn_opt)
        # log.write(eval_data_log)
        # print(f'{accuracy_by_best_model:0.3f}')
        # log.write(f'{accuracy_by_best_model:0.3f}\n')
        # log.close()

        return pred