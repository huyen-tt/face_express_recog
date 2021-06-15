import warnings
warnings.filterwarnings("ignore")
import cv2,os,argparse,torch,random,glob
import numpy as np
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
from face_detect.data import cfg_mnet, cfg_re50
from face_detect.layers.functions.prior_box import PriorBox
from face_detect.utils.nms.py_cpu_nms import py_cpu_nms
from face_detect.models.retinaface import RetinaFace
from face_detect.utils.box_utils import decode, decode_landm
from face_detect.utils.timer import Timer
from face_expression.RAF_DB import image_utils
from face_expression.RAF_DB import Networks


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='face_detect/weights/Resnet50_epoch_95.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--checkpoint', type=str, default='face_expression/RAF_DB/models/RAF-DB/best_model.pth', help='Checkpoint filepath')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
args = parser.parse_args()


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_detect_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def expression(i):
    switcher={
              0:'Surprise',
              1:'Fear',
              2:'Disgust',
              3:'Happy',
              4:'Sad',
              5:'Angry',
              6:'Neutral'
            }
    predict=switcher.get(i,lambda :"Invalid")  
    return predict
    
class FaceDataSet(data.Dataset):
    def __init__(self, img_name, transform=None, basic_aug=False):
        self.raf_path = img_name
        self.transform = transform

        self.file_paths = []
        paths = glob.glob('image/face_img/*{}_*'.format(img_name))
        self.file_paths = paths
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]            # BGR to RGB

        if self.transform is not None:
            image = self.transform(image)
        
        return image, idx
        
    
def detect_face(image_path, image_name):

    cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_detect_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading face detection model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # Detect face
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    target_size = 1600
    max_size = 2150
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
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()


    bboxs = dets
    list_faces = []
    i = 1
    for box in bboxs:
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        confidence = float(box[4])

        if(confidence >= 0.1): 
            i+=1
            list_faces.append([x,y,w,h])
            crop = img_raw[y:y+h, x:x+w]
            crop = cv2.resize(crop, (48, 48))
            cv2.imwrite("image/face_img/{}_{}.jpg".format(img_name,i), crop)
    return list_faces
    
    
def face_expression(img_name, list_faces):
    model = Networks.ResNet18_ARM___RAF()
    checkpoint = torch.load(args.checkpoint)
    model_state_dict = model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)
    print('Finished loading facial expression recognition model!')
        
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    face_dataset = FaceDataSet(img_name=img_name,transform=data_transforms)
    face_num = face_dataset.__len__()
    print('Num of faces:', face_num)
    
    face_loader = torch.utils.data.DataLoader(face_dataset,
                                               batch_size=face_num,
                                               num_workers=4,
                                               shuffle=False,
                                               pin_memory=True)

    model = model.cuda()
    model.eval()
    for img_i, (imgs,_) in enumerate(face_loader):
        output, _ = model(imgs.cuda())
        _, predicts = torch.max(output, 1)
    predicts = predicts.cpu().numpy()
    
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    i=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in list_faces:
        cv2.rectangle(img_raw, (x, y), (x + w, y + h), (0, 255, 0),2)
        exp = expression(predicts[i])
        # cv2.putText(img_raw, exp, (x, y), font, 0.5, (255, 255, 0), 2)
        i+=1
    cv2.imwrite('image/out_img/face_{}.jpg'.format(img_name), img_raw)


if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    image_path = './image/17_Ceremony_Ceremony_17_171.jpg'
    img_name = os.path.basename(image_path)
    img_name = os.path.splitext(img_name)[0]
    list_faces = detect_face(image_path, img_name)
    face_expression(img_name, list_faces)
    
   
    

