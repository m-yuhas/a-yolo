import io
import json
import tempfile
import contextlib
from pycocotools.cocoeval import COCOeval


def COCOEvaluator(json_list, val_dataset):
    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    cocoGt = val_dataset.coco
    # pycocotools box format: (x1, y1, w, h)
    annType = ["segm", "bbox", "keypoints"]

    if len(json_list) > 0:
        _, tmp = tempfile.mkstemp()
        json.dump(json_list, open(tmp, "w"), skipkeys=True, ensure_ascii=True)
        cocoDt = cocoGt.loadRes(tmp)

        coco_pred = {"images": [], "categories": []}
        for (k, v) in cocoGt.imgs.items():
            coco_pred["images"].append(v)
        for (k, v) in cocoGt.cats.items():
            coco_pred["categories"].append(v)
        coco_pred["annotations"] = json_list
        # json.dump(coco_pred, open("./COCO_val.json", "w"))

        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.params.useCats = 0 # MJY

        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = redirect_string.getvalue()

        ### PER CLASS TEST
        for catId in cocoGt.getCatIds():
            rs = io.StringIO()
            with contextlib.redirect_stdout(rs):
                coco_eval = COCOeval(cocoGt, cocoDt, annType[1])
                coco_eval.params.catIds = [catId]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            print(f'Cat {catId}: mAP={coco_eval.stats[0]}, mAP50={coco_eval.stats[1]}')


        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0.0, 0.0, "No detection!"
