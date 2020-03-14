from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import glob

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/RBF_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
# 이미지 저장할 폴더 경로
parser.add_argument('-o', '--save_folder', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# 이미지 파일 불러올 경로
parser.add_argument('-d', '--dataset_folder', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# detect한 것 중 top k 개만 출력함
parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
# 얼굴 크롭한 이미지 안에서 얼굴에 5개 점찍어서 저장
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# 프레임에서 얼굴 detect한 후 크롭해서 저장
parser.add_argument('-c', '--save_face_crop', action="store_true", default=False, help='show detection results')
# 얼굴 크롭한 이미지에서 이마, 눈, 오른쪽눈, 왼쪽눈, 코, 입, 턱 잘라서 저장
parser.add_argument('-p', '--save_face_part', action="store_true", default=False, help='show detection results')
# detect 시간 .csv 파일로 저장
parser.add_argument('-t', '--save_detect_time', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    #print('dataset_folder', args.dataset_folder)
    test_dataset = os.listdir(args.dataset_folder)
    num_images = len(test_dataset)

    #print('test_dataset', test_dataset, 'num_images', num_images)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = args.dataset_folder + img_name
       # print('image_path', image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        org_height, org_width, channel = img_raw.shape
        img = np.float32(img_raw)

        # testing scale
        target_size = args.long_side
        max_size = args.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # detect한 것 전부 출력
        #order = scores.argsort()[::-1]
        # detect한 것 중 top k 개만 출력
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        # detect한 것 전부 출력
        #dets = dets[keep, :]
        #landms = landms[keep]

        # detect한 것 중 top k 개만 출력
        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        # 얼굴 좌표 텍스트로 저장하는 코드
        #save_name = args.save_folder + img_name[:-4] + ".txt"
        #dirname = os.path.dirname(save_name)
        #if not os.path.isdir(dirname):
        #    os.makedirs(dirname)
        #with open(save_name, "w") as fd:
        #    bboxs = dets
        #    file_name = os.path.basename(save_name)[:-4] + "\n"
        #    bboxs_num = str(len(bboxs)) + "\n"
        #    fd.write(file_name)
        #    fd.write(bboxs_num)
        #    for box in bboxs:
        #        x = int(box[0])
        #        y = int(box[1])
        #        w = int(box[2]) - int(box[0])
        #        h = int(box[3]) - int(box[1])
        #        confidence = str(box[4])
        #        line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
        #        fd.write(line)

        #--------------------------------------------------------------------
        # detect시간 text로 저장
        if args.save_detect_time:
            save_detect_time = "C:/Users/Public/facebook/crop/detect_result2.txt"
            dirname = os.path.dirname(save_detect_time)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_detect_time, "a") as fd:
                line = 'im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time) + " \n"
                fd.write(line)

        #print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # --------------------------------------------------------------------
        # 얼굴에 왼쪽 눈, 오른쪽눈, 코끝, 입꼬리 양쪽 다섯 포인트 점찍어서 저장
        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue

                save_5keypoint = img_raw.copy()
                #얼굴에 박스치고 퍼센트 표시
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(save_5keypoint, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(save_5keypoint, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                # 왼쪽 눈, 오른쪽눈, 코끝, 왼쪽 입꼬리, 오른쪽 입꼬리 순으로 색 구분
                cv2.circle(save_5keypoint, (b[5], b[6]), 1, (0, 0, 255), 4)
               # cv2.imshow('left_eye',img_raw)
                cv2.circle(save_5keypoint, (b[7], b[8]), 1, (0, 255, 255), 4)
               # cv2.imshow('right_eye', img_raw)
                cv2.circle(save_5keypoint, (b[9], b[10]), 1, (255, 0, 255), 4)
               # cv2.imshow('nose', img_raw)
                cv2.circle(save_5keypoint, (b[11], b[12]), 1, (0, 255, 0), 4)
               # cv2.imshow('left_mouth', img_raw)
                cv2.circle(save_5keypoint, (b[13], b[14]), 1, (255, 0, 0), 4)
               # cv2.imshow('right_mouth', img_raw)
               # cv2.waitKey(0)
            # save image
            if not os.path.exists(args.save_folder+'detect_raw/'):
                os.makedirs(args.save_folder+'detect_raw/')
            img_file_name = args.save_folder+'detect_raw/' + img_name + ".jpg"
            cv2.imwrite(img_file_name, save_5keypoint)

            # --------------------------------------------------------------------
            # 얼굴 detect한 영역만큼 크롭후 저장
        if args.save_face_crop:

            face_crop = img_raw.copy()
            x1 = b[0]
            x2 = b[2]
            y1 = b[1]
            y2 = b[3]

            height = int(y2) - int(y1)
            width = int(x2) - int(x1)

            plus_rate = 0.1
            height_plus = plus_rate * height
            width_plus = plus_rate * width

            crop_y1 = 0
            crop_y2 = org_height
            crop_x1 = 0
            crop_x2 = org_width

            if y1 - height_plus > 0:
                crop_y1 = int(y1 - height_plus)
            if y2 + height_plus < org_height:
                crop_y2 = int(y2 + height_plus)
            if x1 - width_plus > 0:
                crop_x1 = int(x1 - width_plus)
            if x2 + width_plus < org_width:
                crop_x2 = int(x2 + width_plus)

            crop_img = face_crop[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imshow(crop_img)

          #  if not os.path.exists(args.save_folder+'face_crop/'):
          #      os.makedirs(args.save_folder+'face_crop/')
          #  face_crop = args.save_folder+'face_crop/' + img_name + ".jpg"
          #  cv2.imwrite(face_crop, crop_img)

        # --------------------------------------------------------------------
        # 얼굴 부분별로 저장 왼쪽 눈, 오른쪽 눈, 코, 입
        # 프레임에서 이미지 크롭후 저장할거면 밑에있는 w,h 바꿔야함
        if args.save_face_part:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                save_face_part = img_raw.copy()

                h, w, c = save_face_part.shape

                print('img_name', img_name)

                # 점이 몰려서 좌표가 -일 때 가 있는것 같음
                # 이마, 왼쪽 눈, 오른쪽눈, 코, 입, 턱
                # 좌표 정리
                # 왼쪽 눈 b[5], b[6] 오른쪽 눈 b[7], b[8] 코끝 b[9], b[10] 왼쪽 입꼬리 b[11], b[12] 오른쪽 입꼬리 b[13], b[14]
                #[높이 최소 : 높이 최대, 넓이 최소 : 넓이 최대]

                #이마
                if b[6] > b[8] :
                    forehead_img = save_face_part[0:int(b[6]), 0:w]
                else :
                    forehead_img = save_face_part[0:int(b[8]), 0:w]
                #cv2.imshow('forehead', forehead_img)

                if not os.path.exists(args.save_folder+'forehead/'):
                     os.makedirs(args.save_folder+'forehead/')
                forehed_name = args.save_folder+'forehead/' + img_name + ".jpg"
                cv2.imwrite(forehed_name, forehead_img)


                # 얼굴 좌우로 기울어져있을때 별로임
                # 코끝 점이 한쪽으로 치우쳐져 있을 때 눈이 잘릴 위험성이 있음

                #양쪽 눈
                eye_img = save_face_part[int(b[6]-(b[10]-b[6])):int(b[10]), 0:w]
                #cv2.imshow('eye', eye_img)

                if not os.path.exists(args.save_folder+'eye/'):
                     os.makedirs(args.save_folder+'eye/')
                eye_name = args.save_folder+'eye/' + img_name + ".jpg"
                cv2.imwrite(eye_name, eye_img)

                #왼쪽 눈
                lefteye_img = save_face_part[int(b[6]-(b[10]-b[6])):int(b[10]), 0:int(b[9])]
                #cv2.imshow('lefteye', lefteye_img)

                if not os.path.exists(args.save_folder+'lefteye/'):
                     os.makedirs(args.save_folder+'lefteye/')
                lefteye_name = args.save_folder+'lefteye/' + img_name + ".jpg"
                cv2.imwrite(lefteye_name, lefteye_img)

                #오른쪽 눈
                rightteye_img = save_face_part[int(b[8]-(b[10]-b[8])):int(b[10]):,int(b[9]):w]
                #cv2.imshow('righteye', rightteye_img)

                if not os.path.exists(args.save_folder+'rightteye/'):
                     os.makedirs(args.save_folder+'rightteye/')
                rightteye_name = args.save_folder+'rightteye/' + img_name + ".jpg"
                cv2.imwrite(rightteye_name, rightteye_img)

                #코
                if b[6] < b[8] :
                    nose_img = save_face_part[int(b[6]-(b[10]-b[6])):int(b[12]), int(b[5]):int(b[7])]
                else :
                    nose_img = save_face_part[int(b[8]-(b[10]-b[8])):int(b[12]), int(b[5]):int(b[7])]
                #cv2.imshow('nose', nose_img)

                if not os.path.exists(args.save_folder+'nose/'):
                     os.makedirs(args.save_folder+'nose/')
                nose_name = args.save_folder+'nose/' + img_name + ".jpg"
                cv2.imwrite(nose_name, nose_img)

                #입
                mouth_img = save_face_part[int(b[10]):h, 0:w]
                #cv2.imshow('mouth', mouth_img)

                if not os.path.exists(args.save_folder+'mouth/'):
                     os.makedirs(args.save_folder+'mouth/')
                mouth_name = args.save_folder+'mouth/' + img_name + ".jpg"
                cv2.imwrite(mouth_name, mouth_img)

                #턱
                if b[12] > b[14] :
                    jar_img = save_face_part[int(b[14]):h, 0:w]
                else :
                    jar_img = save_face_part[int(b[12]):h, 0:w]
                #cv2.imshow('jar', jar_img)

                if not os.path.exists(args.save_folder+'jar/'):
                     os.makedirs(args.save_folder+'jar/')
                jar_name = args.save_folder+'jar/' + img_name + ".jpg"
                cv2.imwrite(jar_name, jar_img)

                #cv2.waitKey(0)

