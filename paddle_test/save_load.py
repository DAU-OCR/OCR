# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

from __future__ import absolute_import, division, print_function

import os
import pickle
import json
import errno
import paddle

from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

try:
    import encryption
    encrypted = encryption.is_encryption_needed()
except ImportError:
    get_logger().warning("encryption 모듈을 불러오지 못했습니다. 암호화는 적용되지 않습니다.")
    encrypted = False

__all__ = ["load_model", "save_model"]

def _mkdir(path, logger):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST or not os.path.isdir(path):
                raise

def _convert_config(config):
    try:
        from omegaconf import OmegaConf
        if isinstance(config, dict):
            return config
        if isinstance(config, OmegaConf):
            return OmegaConf.to_container(config, resolve=True)
        return config  # BaseModel 등은 그대로 반환
    except Exception as e:
        print(f"[경고] config 변환 실패: {e}")
        return config



def _load_optimizer_state(optimizer, path, logger):
    if os.path.exists(path + ".pdopt"):
        optim_dict = paddle.load(path + ".pdopt")
        optimizer.set_state_dict(optim_dict)
    else:
        logger.warning(f"{path}.pdopt 파일이 존재하지 않아 optimizer state는 로드되지 않습니다.")

def load_model(config, model, optimizer=None, model_type="det"):
    logger = get_logger()

    # ✅ config가 이상할 경우를 대비해 방어적 처리
    config = _convert_config(config)
    if not isinstance(config, dict):
        logger.warning("config가 dict가 아님. 변환 실패로 그대로 진행합니다.")

    global_cfg = config.get("Global", {})
    checkpoints = global_cfg.get("checkpoints")
    pretrained = global_cfg.get("pretrained_model")
    best_model = {}
    is_fp16 = False

    is_kie = model_type == "kie" and config.get("Architecture", {}).get("algorithm") not in ["SDMGR"]
    if is_kie:
        checkpoints = config["Architecture"]["Backbone"].get("checkpoints", None)
        if checkpoints and os.path.exists(os.path.join(checkpoints, "metric.states")):
            with open(os.path.join(checkpoints, "metric.states"), "rb") as f:
                states = pickle.load(f, encoding="latin1")
            best_model = states.get("best_model_dict", {})
            best_model["start_epoch"] = states.get("epoch", -1) + 1
            logger.info(f"{checkpoints}에서 KIE 모델 로드 완료.")
            if optimizer:
                _load_optimizer_state(optimizer, checkpoints, logger)
        return best_model

    # Detection or Recognition
    if checkpoints:
        ckpt_path = checkpoints.replace(".pdparams", "") if checkpoints.endswith(".pdparams") else checkpoints
        assert os.path.exists(ckpt_path + ".pdparams"), f"{ckpt_path}.pdparams가 존재하지 않습니다!"

        params = paddle.load(ckpt_path + ".pdparams")
        model_dict = model.state_dict()
        loaded_params = {}

        for k, v in model_dict.items():
            if k in params:
                val = params[k]
                is_fp16 |= val.dtype == paddle.float16
                if val.dtype != v.dtype:
                    val = val.astype(v.dtype)
                if val.shape == v.shape:
                    loaded_params[k] = val
                else:
                    logger.warning(f"{k}의 shape 불일치: {v.shape} vs {val.shape}")
            else:
                logger.warning(f"{k}는 체크포인트에 없습니다.")

        model.set_state_dict(loaded_params)
        logger.info(f"{ckpt_path}에서 모델 로드 완료.")

        if optimizer:
            _load_optimizer_state(optimizer, ckpt_path, logger)

        if os.path.exists(ckpt_path + ".states"):
            with open(ckpt_path + ".states", "rb") as f:
                states = pickle.load(f, encoding="latin1")
            best_model = states.get("best_model_dict", {})
            best_model["acc"] = 0.0
            best_model["start_epoch"] = states.get("epoch", -1) + 1

    elif pretrained:
        is_fp16 = load_pretrained_params(model, pretrained)
    else:
        logger.info("신규 학습: 사전 학습된 모델 없음.")

    best_model["is_float16"] = is_fp16
    return best_model


def load_pretrained_params(model, path):
    logger = get_logger()
    path = maybe_download_params(path).replace(".pdparams", "")
    assert os.path.exists(path + ".pdparams"), f"{path}.pdparams가 존재하지 않습니다!"

    params = paddle.load(path + ".pdparams")
    model_dict = model.state_dict()
    new_state = {}
    is_fp16 = False

    for k, v in model_dict.items():
        if k in params:
            param = params[k]
            is_fp16 |= param.dtype == paddle.float16
            if param.dtype != v.dtype:
                param = param.astype(v.dtype)
            if param.shape == v.shape:
                new_state[k] = param
            else:
                logger.warning(f"{k} shape mismatch: model {v.shape}, pretrained {param.shape}")
        else:
            logger.warning(f"{k}는 사전 학습된 모델에 없습니다.")

    model.set_state_dict(new_state)
    logger.info(f"{path}에서 사전 학습된 모델 로드 완료.")
    return is_fp16

def save_model(model, optimizer, model_path, logger, config, is_best=False, prefix="ppocr", **kwargs):
    _mkdir(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    paddle.save(model.state_dict(), model_prefix + ".pdparams")

    if prefix == "best_accuracy":
        best_path = os.path.join(model_path, "best_model")
        _mkdir(best_path, logger)
        paddle.save(optimizer.state_dict(), os.path.join(best_path, "model.pdopt"))
        paddle.save(model.state_dict(), os.path.join(best_path, "model.pdparams"))

    with open(model_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)

    if kwargs.get("save_model_info"):
        with open(os.path.join(model_path, f"{prefix}.info.json"), "w") as f:
            json.dump(kwargs, f)
        logger.info(f"모델 정보가 {model_path}에 저장되었습니다.")

    logger.info(f"{'최고 정확도 모델' if is_best else '모델'} 저장 완료: {model_prefix}")

# 필요 시 update_train_results 함수도 동일하게 정리 가능
