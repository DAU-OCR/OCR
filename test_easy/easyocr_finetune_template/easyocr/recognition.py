
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import importlib  # ✅ 이 줄이 반드시 필요

from easyocr.utils import CTCLabelConverter

def get_recognizer(recog_network, network_params, character,                   separator_list, dict_list, model_path,                   device = 'cpu', quantize = True):

    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)

    if recog_network == 'generation1':
        model_pkg = importlib.import_module("easyocr.model.model")
    elif recog_network == 'generation2':
        model_pkg = importlib.import_module("easyocr.model.vgg_model")
    else:
        model_pkg = importlib.import_module("easyocr." + recog_network)
    model = model_pkg.Model(num_class=num_class, **network_params)

    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if not k.startswith("Prediction."):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    if device != 'cpu':
        model = torch.nn.DataParallel(model).to(device)
    else:
        quantize = False  # GPU에서 양자화된 LSTM은 지원되지 않음


    return model, converter

def get_text(character, imgH, imgW, recognizer, converter, image_list,\
             ignore_char = '',decoder = 'greedy', beamWidth =5, batch_size=1, contrast_ths=0.1,\
             adjust_contrast=0.5, filter_ths = 0.003, workers = 1, device = 'cpu'):
    batch_max_length = int(imgW/10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try: ignore_idx.append(character.index(char)+1)
        except: pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True)

    # predict first round
    result1 = recognizer_predict(recognizer, converter, test_loader,batch_max_length,\
                                 ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    # predict second round
    low_confident_idx = [i for i,item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False,
                        num_workers=int(workers), collate_fn=AlignCollate_contrast, pin_memory=True)
        result2 = recognizer_predict(recognizer, converter, test_loader, batch_max_length,\
                                     ignore_idx, char_group_idx, decoder, beamWidth, device = device)

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1]>pred2[1]:
                result.append( (box, pred1[0], pred1[1]) )
            else:
                result.append( (box, pred2[0], pred2[1]) )
        else:
            result.append( (box, pred1[0], pred1[1]) )

    return result


