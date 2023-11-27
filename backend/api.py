import gc
import math
from typing import Optional, List


from .yolo import getYOLOOutput, extractROI, getYoloNet


def try_get_roi(frame) -> Optional[List]:
    net = getYoloNet()
    dfg, pc = getYOLOOutput(frame, net)
    gc.collect()
    try:
        assert len(dfg) >= 2
        if len(dfg) > 2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg)-1):
                for j in range(i+1, len(dfg)):
                    d = math.sqrt(pow(dfg[i][0]-dfg[j][0], 2) +
                                  pow(dfg[i][1]-dfg[j][1], 2))
                    if d > maxD:
                        tmpdfg = [dfg[i], dfg[j]]
                        maxD = d
            dfg = tmpdfg
        pc = sorted(pc, key=lambda x: x[-1], reverse=True)
        roi = extractROI(frame, dfg, pc)
        return roi
    except:
        return None


def roi_to_embeddings(roi) -> List:
    # TODO
    return []
