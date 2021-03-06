import sys

sys.path.append("/home/hezoujie/Competition/Pytorch_NER")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
import pandas as pd

"""
获取融合结果的后处理结果
"""

config = Config()
data_dir = config.output


def op(row):
    if str(row) == 'nan':
        row = 'DELETE'
    return row


def mark_op(entity_list):
    """
    注意事项：后处理之后删除单字实体
    """

    """
    场景1：extra_chars = set("!,:@")  # 直接干掉
    """
    extra_chars = set("!,:@")
    flag = True
    for i, label in enumerate(entity_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li:
                        flag = False
                        print(li)
                        temp.append("DELETE")
                        break
                if flag:
                    temp.append(li)
                flag = True
            entity_list[i] = ";".join(temp)

    """
    场景2：extra_chars = set("-")  # 在头or尾直接舍弃
    """
    flag = True
    extra_chars = set("-")
    for i, label in enumerate(entity_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                if li in ['远程控制f/a-18_"超级大黄蜂"', 'p-8a_"海神_"反潜机']:
                    temp.append(li)
                else:
                    for ec in extra_chars:
                        if ec in li[0] or ec in li[-1]:
                            flag = False
                            print(li)
                            temp.append("DELETE")
                            break
                    if flag:
                        temp.append(li)
                    flag = True
            entity_list[i] = ";".join(temp)

    """
    场景3：extra_chars = set("()（）‘’“”")   # 若不是对应匹配的括号，括号半边在头与尾，替换成‘’，括号在实体中间则舍弃
    """
    for i, label in enumerate(entity_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                if li in ['f/a-18e/f"超级大黄蜂"战机', '雅典娜"激光武器系统样机', '宙斯盾"弹道导弹防御系统', 'lockib型"标准"-3导弹', '大气层外杀伤器(ekv',
                          '防空反导雷达(spy-6', '小型核动力系统"kilopower"', 'agm-84n"鱼叉"blockⅱ空射型反舰巡航导弹', 'agm-84d"鱼叉"blockⅰc导弹',
                          'f/a-18e/f"超级大黄蜂"战斗机', 'f/a-18e/f"超级大黄蜂"战机升级软件', 'rs-28"萨尔玛特"洲际弹道导弹', 'kh-47m2"匕首"高超声速导弹',
                          'f/a-18f"超级大黄蜂"战斗机', 'rgm-84d"鱼叉"blockc型反舰导弹', 'f-35c"闪电"ⅱ型联合攻击机', 'blockiia型"标准"-3导弹']:
                    temp.append(li)
                else:
                    if '(' not in li and ')' not in li and '（' not in li and '）' not in li and '"' not in li and "'" not in li and '‘' not in li and '’' not in li:
                        temp.append(li)
                    else:
                        flag1, flag2, flag3, flag4, flag5 = True, True, True, True, True

                        if '(' in li or ')' in li:
                            if '(' in li and ')' in li and li.index('(') < li.index(')'):
                                pass
                            else:
                                flag1 = False

                        if '（' in li or '）' in li:
                            if '（' in li and '）' in li and li.index('（') < li.index('）'):
                                pass
                            else:
                                flag2 = False

                        if '‘' in li or '’' in li:
                            if '‘' in li and '’' in li and li.index('‘') < li.index('’'):
                                pass
                            else:
                                flag4 = False

                        if '"' in li:
                            num = li.find('"')
                            if num % 2 == 0:
                                pass
                            else:
                                flag5 = False

                        if flag1 and flag2 and flag3 and flag4 and flag5:
                            temp.append(li)
                        else:
                            print(li)
                            temp.append("DELETE")

            entity_list[i] = ";".join(temp)
    """
       场景4：addition_chars = ['长','重','深','高','快','俄','翼','火','烟','车','印','美']  # 单字不在列表中直接干掉
    """
    addition_chars = ['长', '重', '深', '高', '快', '俄', '翼', '火', '烟', '车', '印', '美']
    for i, label in enumerate(entity_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                if len(li) == 1:
                    if li not in addition_chars:
                        print(li)  # 显示被删除标签
                        temp.append("DELETE")
                    else:
                        temp.append(li)
                else:
                    temp.append(li)

            entity_list[i] = ";".join(temp)

    """
    场景5：extra_chars = set("_")  # 在头后接“，否则舍弃；在尾前跟)，否则直接舍弃
    """
    flag = True
    extra_chars = set("_")
    for i, label in enumerate(entity_list):
        # print(label)
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                if li != "":
                    for ec in extra_chars:
                        if ec in li[0]:
                            if li[1] != '"':
                                flag = False
                                break
                        if ec in li[-1]:
                            if li[-2] != ")":
                                flag = False
                                break
                    if flag:
                        temp.append(li)
                    flag = True
                else:
                    print(li)  # 显示被删除标签
                    temp.append("DELETE")

            entity_list[i] = ";".join(temp)
    # print(entity_list)

    return entity_list


if __name__ == '__main__':
    final_result = pd.read_csv(data_dir + '/test_result.csv', encoding='utf-8')
    final_result['entities'] = final_result['entities'].apply(op)

    entity_list = final_result['entities'].tolist()
    entity_list = mark_op(entity_list)  # 规则-对实体的符号进行处理
    final_result['entities'] = entity_list
    final_result.to_csv(data_dir + '/last_result.csv', index=False, encoding='utf-8')
    print('post_ensemble_result  Done...')
