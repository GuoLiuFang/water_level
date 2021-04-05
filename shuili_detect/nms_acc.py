# coding=utf8
import numpy as np
import cfg


class PredictAfterNms(object):

    def __init__(self):
        pass

    def nms(self, predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
        region_list = self._region_list(activation_pixels)
        D = self._region_group(region_list)  # 这里合同一个文本区域的不同文本行
        quad_list = np.zeros((len(D), 4, 2))
        score_list = np.zeros((len(D), 4))
        for group, g_th in zip(D, range(len(D))):
            total_score = np.zeros((4, 2))
            # 求出该group的 最大连续头像素x 和 最小连续尾像素x
            headx_list = []
            tailx_list = []
            x_list = []
            for row in group:
                # 前后边界分开来写
                # 先是第一个边界
                for ij in region_list[row]:
                    # 在这里加上一个break 解决前后merge问题
                    score = predict[ij[0], ij[1], 1]
                    x_list.append(ij[1])
                    if score >= threshold:
                        ith_score = predict[ij[0], ij[1], 2:3]
                        # 说明是头像素
                        if ith_score < cfg.trunc_threshold:
                            headx_list.append(ij[1])
                        # 说明是尾像素
                        if ith_score >= 1 - cfg.trunc_threshold:
                            tailx_list.append(ij[1])
            headx_list = sorted(headx_list)
            tailx_list = sorted(tailx_list, reverse=True)
            min_tail_x = 0
            max_head_x = 20000
            if len(headx_list) > 2:
                max_head_x = headx_list[0]
                for i in range(1, len(headx_list)):
                    if headx_list[i] - headx_list[i-1] <= 1:
                        max_head_x = max(max_head_x, headx_list[i])
                    else:
                        break
            if len(tailx_list) > 2:
                min_tail_x = tailx_list[0]
                for i in range(1, len(tailx_list)):
                    if tailx_list[i-1] - tailx_list[i] <= 1:
                        min_tail_x = min(tailx_list[i], min_tail_x)
                    else:
                        break

            for row in group:
                # 前后边界分开来写
                # 先是第一个边界
                for ij in region_list[row]:
                    # 在这里加上一个break 解决前后merge问题
                    score = predict[ij[0], ij[1], 1]
                    if score >= threshold:
                        ith_score = predict[ij[0], ij[1], 2:3]
                        # 说明是头像素
                        if ith_score < cfg.trunc_threshold and ij[1] <= max_head_x:
                            ith = int(np.around(ith_score))
                            total_score[ith * 2:(ith + 1) * 2] += score
                            px = (ij[1] + 0.5) * cfg.pixel_size
                            py = (ij[0] + 0.5) * cfg.pixel_size
                            p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                            quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
                        # 说明是尾像素
                        if ith_score >= 1 - cfg.trunc_threshold and ij[1] >= min_tail_x:
                            ith = int(np.around(ith_score))
                            total_score[ith * 2:(ith + 1) * 2] += score
                            px = (ij[1] + 0.5) * cfg.pixel_size
                            py = (ij[0] + 0.5) * cfg.pixel_size
                            p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                            quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v

            score_list[g_th] = total_score[:, 0]
            quad_list[g_th] /= (total_score + cfg.epsilon)
        return score_list, quad_list

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
