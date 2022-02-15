from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics 
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
import src.utils.converter as converter
import src.utils.general_utils as general_utils
import concurrent.futures
import time

dir_annotations_gt = '/home/kuluum/rublevo_markup/gt/all'
dir_images_gt = '/home/kuluum/rublevo_markup/gt/all_img'
filepath_classes = '/home/kuluum/rublevo_markup/names.names'
dir_dets = '/home/kuluum/rublevo_markup/simple_det/all'
# self.filepath_classes_det = '/home/kuluum/rublevo_markup/names.names'
# self.dir_save_results = '/home/kuluum/rublevo_markup/simple_det'


start = time.time()

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    gt_annotations_future = executor.submit(converter.yolo2bb,
                                    dir_annotations_gt,
                                    dir_images_gt,
                                    filepath_classes,
                                    bb_type=BBType.GROUND_TRUTH)


    det_annotations_future = executor.submit(converter.text2bb,
                                        dir_dets,
                                        bb_type=BBType.DETECTED,
                                        bb_format=BBFormat.XYWH,
                                        type_coordinates=CoordinatesType.RELATIVE,
                                        img_dir=dir_images_gt)

    concurrent.futures.wait([gt_annotations_future, det_annotations_future])
    gt_annotations = gt_annotations_future.result()
    det_annotations = det_annotations_future.result()

det_annotations = general_utils.replace_id_with_classes(det_annotations, filepath_classes)

pascal_res = get_pascalvoc_metrics(gt_annotations,
                                    det_annotations,
                                    iou_threshold=0.5,
                                    generate_table=False)

mAP = pascal_res['mAP']

end = time.time()
print(mAP)
print(f'time: {end - start} sec')