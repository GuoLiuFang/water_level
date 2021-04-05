# coding=utf8
import numpy as np
import cfg


class PredictAfterNms(object):

    def __init__(self):
        pass
    # todo 增加hard nms
    def nms(self, predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
        region_list = self._region_list(activation_pixels)
        D = self._region_group(region_list)  # 这里合同一个文本区域的不同文本行
        bboxes = []
        for group, g_th in zip(D, range(len(D))):
            # 这里一个group就是一个表格区域了
            # 求出该group的 最大连续头像素x 和 最小连续尾像素x
            x_list = []
            y_list = []
            for row in group:
                # 前后边界分开来写
                # 先是第一个边界
                for ij in region_list[row]:
                    # 在这里加上一个break 解决前后merge问题
                    # score = predict[ij[0], ij[1], 0]  # 第0位是是否是文本区域的置信度
                    x_list.append(ij[1])  # x是ij[1] 则 y是ij[0]
                    y_list.append(ij[0])

            min_x = min(x_list)
            max_x = max(x_list)
            min_y = min(y_list)
            max_y = max(y_list)

            # 计算置信度
            scores = []
            for x in range(min_x, max_x+1):
                for y in range(min_y, max_y+1):
                    score = predict[x, y, 0]
                    scores.append(score)
            score = np.mean(scores)
            bboxes.append([min_x, max_x, min_y, max_y, score])

        # 这里进行了hard nms
        bboxes_nmsed = self.area_nms(bboxes)
        quad_list = np.zeros((len(bboxes_nmsed), 4, 2))
        score_list = np.zeros((len(bboxes_nmsed), 1))
        for i in range(len(bboxes_nmsed)):
            min_x, max_x, min_y, max_y, score = bboxes_nmsed[i]
            min_x, max_x, min_y, max_y = np.clip((min_x - 2, max_x + 2, min_y - 2, max_y + 2), 0, predict.shape[0])
            quad_list[i] = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]) * cfg.pixel_size
            score_list[i] = score

        return score_list, quad_list

    # 利用面积做nms
    def area_nms(self, bounding_boxes, Nt=0.3):
        if len(bounding_boxes) == 0:
            return []
        bboxes = np.array(bounding_boxes)

        # 计算 n 个候选框的面积大小
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 2]
        x2 = bboxes[:, 1]
        y2 = bboxes[:, 3]
        scores = bboxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
        order = np.argsort(areas)

        picked_boxes = []  # 返回值
        while order.size > 0:
            # 将当前置信度最大的框加入返回值列表中
            index = order[-1]
            picked_boxes.append(bounding_boxes[index])

            # 获取当前置信度最大的候选框与其他任意候选框的相交面积
            x11 = np.maximum(x1[index], x1[order[:-1]])
            y11 = np.maximum(y1[index], y1[order[:-1]])
            x22 = np.minimum(x2[index], x2[order[:-1]])
            y22 = np.minimum(y2[index], y2[order[:-1]])
            w = np.maximum(0.0, x22 - x11 + 1)
            h = np.maximum(0.0, y22 - y11 + 1)
            intersection = w * h

            # 利用相交的面积和面积较小的一个的交并比, 将交并比大于阈值的框删除
            iops = intersection / (areas[order[:-1]])
            left = np.where(iops < Nt)
            ori_size = order.size
            order = order[left]
            if order.size < ori_size-1:
                print('*'*20)
        return picked_boxes

    def _region_list(self, activation_pixels):
        region_list = []
        region_list_idx = []
        last_i = -1
        zipv = zip(activation_pixels[0], activation_pixels[1])
        for i, j in zipv:
            if i != last_i:
                region_list.append({(i, j)})
                region_list_idx.append(i)
                last_i = i
                continue
            merge = False
            for k in range(len(region_list)):
                current_i = region_list_idx[k]
                if i != current_i:
                    continue
                if self._should_merge(region_list[k], i, j):
                    region_list[k].add((i, j))
                    merge = True
            if not merge:
                region_list.append({(i, j)})
                region_list_idx.append(i)
        return region_list

    def _should_merge(self, region, i, j):
        neighbor = {(i, j - 1)}
        return not region.isdisjoint(neighbor)

    def _region_group(self, region_list):
        S = [i for i in range(len(region_list))]
        D = []
        while len(S) > 0:
            m = S.pop(0)
            if len(S) == 0:
                D.append([m])
            else:
                D.append(self._rec_region_merge(region_list, m, S))
        return D

    def _rec_region_merge(self, region_list, m, S):
        rows = [m]
        tmp = []
        for n in S:
            # 判断 n > m的目的是：防止n从m后边追上来时，被break，比如：n=44；m=56
            # NMS_TRUNCATE_LOOP,这个值，可能会引入diff；尤其是竖着的文字区域比较多的时候
            if n > m and abs(n - m) > 40:
                break

            if not self._region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                    not self._region_neighbor(region_list[n]).isdisjoint(region_list[m]):
                tmp.append(n)
        for d in tmp:
            S.remove(d)
        for e in tmp:
            rows.extend(self._rec_region_merge(region_list, e, S))
        return rows

    def _region_neighbor(self, region_set):
        j_min = 100000000
        j_max = -1
        i_m = 0
        for node in region_set:
            i_m = node[0] + 1
            if node[1] > j_max:
                j_max = node[1]
            if node[1] < j_min:
                j_min = node[1]
        j_min = j_min - 1
        j_max = j_max + 2
        neighbor = set()
        for j in range(j_min, j_max):
            neighbor.add((i_m, j))
        return neighbor
