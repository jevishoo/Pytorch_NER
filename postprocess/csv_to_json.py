"""
    @author: Jevis-Hoo
    @Date: 2020/5/18 19:58
    @Description: 
"""
import pandas as pd
import numpy as np
import sys
import json

sys.path.append("/home/hezoujie/Competition/Pytorch_NER")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块

from config import Config


def get_label(num):
    if num in [2, 3, 4, 5]:
        return "试验要素"
    elif num in [6, 7, 8, 9]:
        return "性能指标"
    elif num in [10, 11, 12, 13]:
        return "系统组成"
    else:
        return "任务场景"


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 1:
        return index_list
    else:
        return -1


def get_min_num(space, x):
    """
    得到space列表中比 x 小的数的个数
    """
    num = 0
    for s_num in space:
        if s_num <= x:
            num += 1
    return num


def main():
    config = Config()
    test_label = ['validate_V2_1.json',
                  'validate_V2_10.json',
                  'validate_V2_100.json',
                  'validate_V2_11.json',
                  'validate_V2_12.json',
                  'validate_V2_13.json',
                  'validate_V2_14.json',
                  'validate_V2_15.json',
                  'validate_V2_16.json',
                  'validate_V2_17.json',
                  'validate_V2_18.json',
                  'validate_V2_19.json',
                  'validate_V2_2.json',
                  'validate_V2_20.json',
                  'validate_V2_21.json',
                  'validate_V2_22.json',
                  'validate_V2_23.json',
                  'validate_V2_24.json',
                  'validate_V2_25.json',
                  'validate_V2_26.json',
                  'validate_V2_27.json',
                  'validate_V2_28.json',
                  'validate_V2_29.json',
                  'validate_V2_3.json',
                  'validate_V2_30.json',
                  'validate_V2_31.json',
                  'validate_V2_32.json',
                  'validate_V2_33.json',
                  'validate_V2_34.json',
                  'validate_V2_35.json',
                  'validate_V2_36.json',
                  'validate_V2_37.json',
                  'validate_V2_38.json',
                  'validate_V2_39.json',
                  'validate_V2_4.json',
                  'validate_V2_40.json',
                  'validate_V2_41.json',
                  'validate_V2_42.json',
                  'validate_V2_43.json',
                  'validate_V2_44.json',
                  'validate_V2_45.json',
                  'validate_V2_46.json',
                  'validate_V2_47.json',
                  'validate_V2_48.json',
                  'validate_V2_49.json',
                  'validate_V2_5.json',
                  'validate_V2_50.json',
                  'validate_V2_51.json',
                  'validate_V2_52.json',
                  'validate_V2_53.json',
                  'validate_V2_54.json',
                  'validate_V2_55.json',
                  'validate_V2_56.json',
                  'validate_V2_57.json',
                  'validate_V2_58.json',
                  'validate_V2_59.json',
                  'validate_V2_6.json',
                  'validate_V2_60.json',
                  'validate_V2_61.json',
                  'validate_V2_62.json',
                  'validate_V2_63.json',
                  'validate_V2_64.json',
                  'validate_V2_65.json',
                  'validate_V2_66.json',
                  'validate_V2_67.json',
                  'validate_V2_68.json',
                  'validate_V2_69.json',
                  'validate_V2_7.json',
                  'validate_V2_70.json',
                  'validate_V2_71.json',
                  'validate_V2_72.json',
                  'validate_V2_73.json',
                  'validate_V2_74.json',
                  'validate_V2_75.json',
                  'validate_V2_76.json',
                  'validate_V2_77.json',
                  'validate_V2_78.json',
                  'validate_V2_79.json',
                  'validate_V2_8.json',
                  'validate_V2_80.json',
                  'validate_V2_81.json',
                  'validate_V2_82.json',
                  'validate_V2_83.json',
                  'validate_V2_84.json',
                  'validate_V2_85.json',
                  'validate_V2_86.json',
                  'validate_V2_87.json',
                  'validate_V2_88.json',
                  'validate_V2_89.json',
                  'validate_V2_9.json',
                  'validate_V2_90.json',
                  'validate_V2_91.json',
                  'validate_V2_92.json',
                  'validate_V2_93.json',
                  'validate_V2_94.json',
                  'validate_V2_95.json',
                  'validate_V2_96.json',
                  'validate_V2_97.json',
                  'validate_V2_98.json',
                  'validate_V2_99.json']
    final_result = pd.read_csv(config.output + '/last_result.csv', encoding='utf-8')

    f1 = open("./validate_data.json", encoding='utf-8')
    data = json.load(f1)

    space_list = []
    for key in data.keys():
        space_list.append(find_all(" ", data[key]))

    print(space_list)
    print(len(space_list))
    for index in range(final_result.shape[0]):
        if space_list[index] == -1:
            continue
        print(final_result['end_pos'][index])
        start_pos_list = eval(final_result['start_pos'][index])
        end_pos_list = eval(final_result['end_pos'][index])

        start_pos_list = list(np.sort(start_pos_list))
        end_pos_list = list(np.sort(end_pos_list))

        start_space_list = space_list[index].copy()
        end_space_list = space_list[index].copy()

        for i in range(len(start_pos_list)):
            start_min_num = get_min_num(start_space_list, start_pos_list[i])
            for j in range(i, len(start_pos_list)):
                start_pos_list[j] += start_min_num
            del start_space_list[:start_min_num]

            end_min_num = get_min_num(end_space_list, end_pos_list[i])
            for j in range(i, len(end_pos_list)):
                end_pos_list[j] += end_min_num
            del end_space_list[:end_min_num]

        final_result['start_pos'][index] = str(start_pos_list)
        final_result['end_pos'][index] = str(end_pos_list)

        print(final_result['end_pos'][index])
    out_result_file = pd.Series()

    for index in range(final_result.shape[0]):
        # print(final_result["originalText"][index])
        # print(final_result["entities"][index])
        print(final_result["entities"][index].split(";"))
        label_type_list = eval(final_result["label_type"][index])
        # print(final_result['start_pos'][index])
        # print(final_result["originalText"][index].find(final_result["entities"][index].split(";")[0]))
        start_pos_list = eval(final_result['start_pos'][index])
        print(len(final_result["entities"][index].split(";")))
        end_pos_list = eval(final_result['end_pos'][index])

        entity_list = final_result["entities"][index].split(";")

        result_list = []
        temp_list = []  # 用来记录此条中的entity
        for i in range(len(start_pos_list)):
            if entity_list[i] == "DELETE":
                continue

            if entity_list[i] not in temp_list:  # 用来去除重复预测值
                temp_list.append(entity_list[i])
            else:
                continue

            dic = {'label_type': label_type_list[i], 'overlap': 0}

            if entity_list[i] in ['f/a-18e/f"超级大黄蜂"战机', '雅典娜"激光武器系统样机', '宙斯盾"弹道导弹防御系统', '追梦者”可重复使用航天运载飞行器',
                                  '米格_”-35多用途战斗机']:
                dic['start_pos'] = int(start_pos_list[i])
                dic['end_pos'] = int(end_pos_list[i]) + 1
            elif entity_list[i] in ['大气层外杀伤器(ekv', '防空反导雷达(spy-6']:
                dic['start_pos'] = int(start_pos_list[i]) + 1
                dic['end_pos'] = int(end_pos_list[i]) + 2
            else:
                dic['start_pos'] = int(start_pos_list[i]) + 1
                dic['end_pos'] = int(end_pos_list[i]) + 1

            result_list.append(dic)

        print(len(result_list))
        out_result_file[test_label[index]] = result_list
    out_result_file.to_json(config.checkpoint + "/saved.json",
                            force_ascii=False)  # 本地查看文件
    out_result_file.to_json(config.output + "/validate_1.json - validate_100.json")  # 提交文件


if __name__ == "__main__":
    main()
    print('\n转换成功')
