#!/usr/bin/env python
# coding: utf-8

from konlpy.tag import Okt

import numpy as np
import math
import json
import pandas as pd
from .ModelSingleton import *


def get_similarity(model, word1, word2):
    return model.wv.similarity(word1, word2)


def get_similarities(model, sub_words):
    subw_len = len(sub_words)
    similarities = np.zeros((subw_len, subw_len))
    for i in range(subw_len):
        for j in range(subw_len):
            similarities[i][j] = get_similarity(model, sub_words[i], sub_words[j])
    return similarities


def get_correlation_score(model, word, word_list):
    return np.array([get_similarity(model, word, m) for m in word_list]).mean()


def get_block_score(model, word, center, related_words, lower_similarity_bound):
    block_score = get_similarity(model, word, center)
    block_score += np.array([lower_similarity_bound] + [get_similarity(model, word, m) for m in related_words if
                                                        get_similarity(model, word, m) > lower_similarity_bound]).mean()
    return block_score / 2


def get_center_idx(similarities, words_idx):
    center_idx = words_idx[0]
    max_sum_weight = 0
    for idx in words_idx:
        sum_weight = sum([similarities[idx][m] for m in words_idx])
        if sum_weight > max_sum_weight:
            max_sum_weight = sum_weight
            center_word_idx = idx
    return center_word_idx


##Local input init
def make_word_block(model, sub_words, sub_correlations, min_block_words_num, lower_similarity_bound, low_rate):
    ##Init
    word_block = {}

    # 유사도를 배열에 저장
    similarities = get_similarities(model, sub_words)

    # 글로벌 센터 만들기
    word_weight = np.where(similarities < lower_similarity_bound, similarities * low_rate, similarities).sum(axis=1)
    center_word_idx = np.argmax(word_weight)

    # 글로벌 센터를 제외한 단어를 글로벌 센터와의 연관도순으로 정렬
    relation_words_idx = similarities[center_word_idx].argsort()[::-1][1:]

    ##다른 단어 중 중심단어와의 유사도가 가장 큰 단어 먼저 추가.
    # -연관도가 너무 낮으면 제외.
    must_have_words_idx = [m for m in relation_words_idx if
                           center_word_idx == similarities[m].argsort()[::-1][1] and similarities[center_word_idx][
                               m] > lower_similarity_bound]
    block_words_idx = [center_word_idx] + must_have_words_idx

    # 혹시나 꼭 담아야하는 단어가 8개를 넘길경우 줄이기 (만다라트 제한)
    if (len(block_words_idx) > 9):
        block_words_idx = block_words_idx[:9]

    # 단어가 3개 이상일 경우 센터 단어 재설정1 (글로벌 센터 -> 로컬 센터)
    if len(block_words_idx) > 2:
        block_words_idx.append(center_word_idx)
        center_word_idx = get_center_idx(similarities, block_words_idx)
        block_words_idx.remove(center_word_idx)
    ###########테스트 - 글로벌 센터를 제외한 단어를 글로벌 센터와의 연관도순으로 정렬
    relation_words_idx = similarities[center_word_idx].argsort()[::-1][1:]

    # 클러스터링 안된 단어들 처리
    remain_words_idx = [m for m in relation_words_idx if m not in block_words_idx]

    ##최소 갯수 채울 때까지 클러스터링
    # -연관도가 너무 낮으면 제외.
    while len(remain_words_idx) > 0 and len(must_have_words_idx) + 1 < min_block_words_num:
        candidate_idx = remain_words_idx[0]
        related_words = [str(sub_words[m]) for m in must_have_words_idx]
        score = get_block_score(model, sub_words[candidate_idx], sub_words[center_word_idx], related_words,
                                lower_similarity_bound)
        if score > lower_similarity_bound:
            must_have_words_idx.append(candidate_idx)
            remain_words_idx = remain_words_idx[1:]
            ###########테스트 - 센터 단어 재설정
            must_have_words_idx.append(center_word_idx)
            center_word_idx = get_center_idx(similarities, must_have_words_idx)
            must_have_words_idx.remove(center_word_idx)
        else:
            break

    # 단어가 추가됬으니 센터 단어 재설정2 (로컬 센터 -> new 로컬 센터)
    must_have_words_idx.append(center_word_idx)
    center_word_idx = get_center_idx(similarities, must_have_words_idx)
    must_have_words_idx.remove(center_word_idx)

    word_block['center_word'] = sub_words[center_word_idx]
    word_block['related_words'] = [sub_words[m] for m in must_have_words_idx]
    word_block['related_correlation'] = [sub_correlations[m] for m in must_have_words_idx]

    return (word_block, [sub_words[m] for m in remain_words_idx])


def make_word_block_no_lower_bound(model, sub_words, sub_correlations, min_block_words_num, lower_similarity_bound, low_rate):
    ##Init
    word_block = {}

    # 유사도를 배열에 저장
    similarities = get_similarities(model, sub_words)

    # 글로벌 센터 만들기
    word_weight = np.where(similarities < 1.1, similarities * low_rate, similarities).sum(axis=1)
    center_word_idx = np.argmax(word_weight)
    # 글로벌 센터를 제외한 단어를 글로벌 센터와의 연관도순으로 정렬
    relation_words_idx = similarities[center_word_idx].argsort()[::-1][1:]

    ##다른 단어 중 중심단어와의 유사도가 가장 큰 단어 먼저 추가.
    # -연관도가 너무 낮으면 제외.
    must_have_words_idx = [m for m in relation_words_idx if
                           center_word_idx == similarities[m].argsort()[::-1][1]]
    block_words_idx = [center_word_idx] + must_have_words_idx

    # 혹시나 꼭 담아야하는 단어가 8개를 넘길경우 줄이기 (만다라트 제한)
    if (len(block_words_idx) > 9):
        block_words_idx = block_words_idx[:9]
    # 단어가 3개 이상일 경우 센터 단어 재설정1 (글로벌 센터 -> 로컬 센터)
    if len(block_words_idx) > 2:
        block_words_idx.append(center_word_idx)
        center_word_idx = get_center_idx(similarities, block_words_idx)
        block_words_idx.remove(center_word_idx)
    ###########테스트 - 글로벌 센터를 제외한 단어를 글로벌 센터와의 연관도순으로 정렬
    relation_words_idx = similarities[center_word_idx].argsort()[::-1][1:]
    # 클러스터링 안된 단어들 처리
    remain_words_idx = [m for m in relation_words_idx if m not in block_words_idx]
    ##최소 갯수 채울 때까지 클러스터링
    # -연관도가 너무 낮으면 제외.
    while len(remain_words_idx) > 0 and len(must_have_words_idx) + 1 < 9:
        candidate_idx = remain_words_idx[0]

        must_have_words_idx.append(candidate_idx)
        remain_words_idx = remain_words_idx[1:]
            ###########테스트 - 센터 단어 재설정
        must_have_words_idx.append(center_word_idx)
        center_word_idx = get_center_idx(similarities, must_have_words_idx)
        must_have_words_idx.remove(center_word_idx)

    # 단어가 추가됬으니 센터 단어 재설정2 (로컬 센터 -> new 로컬 센터)
    must_have_words_idx.append(center_word_idx)
    center_word_idx = get_center_idx(similarities, must_have_words_idx)
    must_have_words_idx.remove(center_word_idx)
    print(len(remain_words_idx))
    word_block['center_word'] = sub_words[center_word_idx]
    word_block['related_words'] = [sub_words[m] for m in must_have_words_idx]
    word_block['related_correlation'] = [sub_correlations[m] for m in must_have_words_idx]

    return (word_block, [sub_words[m] for m in remain_words_idx])


def clustering(input_sentence, model_index):
    model = Logger().models[model_index]

    tokenizer = TokenizerLogger().tokenizer
    nouns = input_sentence.split(' ')
    print(nouns)
    input_word_list = nouns

    word_weight = {}
    weighted_word = ''
    for word in input_word_list:
        if word in word_weight.keys():
            word_weight[word] += 1
            weighted_word = word
            break
        else:
            word_weight[word] = 1

    print(word_weight)
    topn_num = 100

    input_word_list = list(set(input_word_list))
    input_length = len(input_word_list)
    extracted_words = []
    extracted_correlation = []
    if len(input_word_list) > 1:
        each_num = topn_num // (input_length-1)
    else:
        each_num = topn_num // input_length * 2
    for input_word in input_word_list:
        if input_word == weighted_word and len(input_word_list) > 1:
            res = model.wv.most_similar(input_word, topn=topn_num)
        else:
            res = model.wv.most_similar(input_word, topn=each_num)
        for m in res:
            extracted_words.extend([m[0]])
            extracted_correlation.extend([m[1]])

    extracted_words = list(set(extracted_words))
    extracted_correlation = list(set(extracted_correlation))

    related_words = []  # input_word_list[:]
    for i in range(len(extracted_words)):
        word = extracted_words[i]
        related_words.append([word, get_correlation_score(model, word, input_word_list), extracted_correlation[i]])
    related_words = sorted(related_words, key=lambda info: info[1])
    related_words.reverse()

    # str_to_idx = {}
    input_words = input_word_list[:]
    input_correlation = [1 for _ in range(len(input_word_list))]
    input_words.extend([m[0] for m in related_words[:]])
    input_correlation.extend([m[2] for m in related_words[:]])

    similarities = get_similarities(model, input_words)
    similarities_mean = similarities.mean() - 1 / len(input_words)

    main_word = input_words[0]

    # Hyper Parameters
    max_words_num = 8
    max_blocks_num = 8  # math.ceil(math.sqrt(len(input_words)))
    lower_bound_rate = 1.1  # 높을수록 평균보다 연관도 높은애끼리 묶음
    lower_similarity_bound = lower_bound_rate * similarities_mean
    low_rate = 0.05

    # local_input : 메인 단어를 제외한 클러스터링할 단어들
    local_input = input_words[input_length:]
    local_correlation = input_correlation[input_length:]

    ##최소 블럭 갯수 설정 (3 <= min_block_words_num <= 8)
    # -중심단어라고 말하려면 최소 3개의 이상의 단어가 있어야함.
    # -만다라트 칸이 최대 8개의 주변단어만 가능
    min_block_words_num = min(max(math.ceil(len(local_input) / max_blocks_num) + 1, 3), max_words_num)
    word_blocks = []
    center_words = []

    # Making word blocks
    for i in range(100):  # max_blocks_num
        if len(local_input) > 2:
            (new_block, local_input) = make_word_block(model, local_input, local_correlation, min_block_words_num,
                                                       lower_similarity_bound, low_rate)
            word_blocks.append(new_block)
        else:
            break

    # Inserting remained words into suit blocks
    for i in range(len(local_input)):
        remained_word = local_input[i]
        remained_correlation = local_correlation[i]
        block_score = np.array(
            [get_block_score(model, remained_word, m['center_word'], m['related_words'], lower_similarity_bound) for m in
             word_blocks])
        word_blocks[block_score.argmax()]['related_words'].append(remained_word)
        word_blocks[block_score.argmax()]['related_correlation'].append(remained_correlation)

    result = {"n_block": len(word_blocks), "blocks": word_blocks}

    clusters = []
    for block in result['blocks']:
        words_list = []
        related_words = [m.split('/')[0] for m in block['related_words']]
        related_correlation = [m for m in block['related_correlation']]
        correlation = get_block_score(model, block['center_word'], input_word_list[0], input_word_list, lower_similarity_bound)
        center_word = block['center_word'].split('/')[0]
        for i in range(len(related_words)):
            word_dict = dict()
            word_dict['text'] = related_words[i]
            word_dict['correlation'] = related_correlation[i]
            t_dict = dict()
            t_dict['word'] = word_dict
            words_list.append(t_dict)
        words_dict = dict()
        words_dict['words'] = words_list
        words_dict['category'] = center_word
        words_dict['correlation'] = correlation
        clusters.append(words_dict)

    return len(clusters), clusters


def make_mandal_art(input_sentence, model_index):
    model = Logger().models[model_index]

    tokenizer = TokenizerLogger().tokenizer
    nouns = input_sentence.split(' ')
    print(nouns)
    input_word_list = nouns

    input_length = len(input_word_list)
    topn_num = 73 - input_length

    extracted_words = []
    extracted_correlation = []
    each_num = topn_num // input_length * 2
    for input_word in input_word_list:
        res = model.wv.most_similar(input_word, topn=each_num)
        for m in res:
            extracted_words.extend([m[0]])
            extracted_correlation.extend([m[1]])

    extracted_words = list(set(extracted_words))
    extracted_correlation = list(set(extracted_correlation))

    related_words = []  # input_word_list[:]
    for i in range(len(extracted_words)):
        word = extracted_words[i]
        related_words.append([word, get_correlation_score(model, word, input_word_list), extracted_correlation[i]])
    related_words = sorted(related_words, key=lambda info: info[1])
    related_words.reverse()

    # str_to_idx = {}
    input_words = input_word_list[:]
    input_correlation = [1 for _ in range(len(input_word_list))]
    input_words.extend([m[0] for m in related_words[:]])
    input_correlation.extend([m[2] for m in related_words[:]])
    input_words = list(set(input_words))
    input_words = input_words[:72]
    input_words.insert(0, input_word_list[0])
    input_correlation = input_correlation[:73]

    similarities = get_similarities(model, input_words)
    similarities_mean = similarities.mean() - 1 / len(input_words)

    main_word = input_words[0]

    # Hyper Parameters
    max_words_num = 8
    max_blocks_num = 8  # math.ceil(math.sqrt(len(input_words)))
    lower_bound_rate = 1.1  # 높을수록 평균보다 연관도 높은애끼리 묶음
    lower_similarity_bound = lower_bound_rate * similarities_mean
    low_rate = 0.05

    # local_input : 메인 단어를 제외한 클러스터링할 단어들
    local_input = input_words[1:]
    local_correlation = input_correlation[1:]

    ##최소 블럭 갯수 설정 (3 <= min_block_words_num <= 8)
    # -중심단어라고 말하려면 최소 3개의 이상의 단어가 있어야함.
    # -만다라트 칸이 최대 8개의 주변단어만 가능
    min_block_words_num = min(max(math.ceil(len(local_input) / max_blocks_num) + 1, 3), max_words_num)
    word_blocks = []
    center_words = []
    print(len(local_input))
    # Making word blocks
    for i in range(max_blocks_num):  # max_blocks_num
        (new_block, local_input) = make_word_block_no_lower_bound(model, local_input, local_correlation, min_block_words_num,
                                                       lower_similarity_bound, low_rate)
        print(new_block['center_word'], new_block['related_words'], len(local_input))
        word_blocks.append(new_block)


    result = []
    result.append(input_words[0].split('/')[0])
    for i in range(max_blocks_num):
        result.append(word_blocks[i]['center_word'].split('/')[0])
    for i in range(max_blocks_num):
        for j in range(max_words_num):
            result.append(word_blocks[i]['related_words'][j].split('/')[0])
    # Inserting remained words into suit blocks

    return result
'''
    clusters = []
    for block in result['blocks']:
        words_list = []
        related_words = [m.split('/')[0] for m in block['related_words']]
        related_correlation = [m for m in block['related_correlation']]
        correlation = get_block_score(model, block['center_word'], input_word_list[0], input_word_list, lower_similarity_bound)
        center_word = block['center_word'].split('/')[0]
        for i in range(len(related_words)):
            word_dict = dict()
            word_dict['text'] = related_words[i]
            word_dict['correlation'] = related_correlation[i]
            t_dict = dict()
            t_dict['word'] = word_dict
            words_list.append(t_dict)
        words_dict = dict()
        words_dict['words'] = words_list
        words_dict['category'] = center_word
        words_dict['correlation'] = correlation
        clusters.append(words_dict)
'''