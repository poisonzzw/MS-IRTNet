
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils.data_loder import testpath,TESTData, test_Daypath,test_Nightpath,TEST_nightData,TEST_dayData,test_visualpath,TEST_visualData
from Convnextv2.builder import Convnextv2

from Utils.metrics import SegEvaluator
import numpy as np
from PIL import Image


def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions):
    palette = get_palette()
    outpath = 'D:/'
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): 
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(outpath + str(image_name)[2:8] + '.png')


class Test(object):
    def __init__(self, sys_args):
        super(Test, self).__init__()
        self.FLAGS = sys_args



    def test_val(self, image_path_a, image_path_b, lable_path):
        # Time
        # start = time.time()

        # Data
        data = TESTData(image_path_a, image_path_b, lable_path)

        # data = TEST_visualData(image_path_a, image_path_b, lable_path)
        # Data_Loader
        test_data = DataLoader(dataset=data, batch_size=1)

        # BUILD Model


        model = torch.load('checkpoint/model.pth').to(self.FLAGS['System_Parameters']['device'])
       
        print("开始加载预训练模型了哦！")

        # pretrain_dict = torch.load('checkpoint/Model_statedict.pth', map_location='cpu')
        # model = Convnextv2(in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(self.FLAGS['System_Parameters']['device'])
        # model_dict = {}
        # mstate_dict = model.state_dict()
        # for k, v in pretrain_dict['state_dict'].items():
        #     if k in mstate_dict:
        #         model_dict[k] = v
        #         print(k)
        # mstate_dict.update(model_dict)
        # model.load_state_dict(mstate_dict)



        metric = SegEvaluator(9, self.FLAGS)

        print(len(test_data))
        with torch.no_grad():
            model.eval()
            metric.reset()
            with tqdm(total=len(test_data), ncols=100, ascii=True) as t:

                for i, (img_ir, img_vis, img_lable, img_name) in enumerate(test_data):
                    t.set_description('||Test-ALL Image %s' % (i + 1))

                    img_ir = img_ir.to(self.FLAGS['System_Parameters']['device'])
                    img_vis = img_vis.to(self.FLAGS['System_Parameters']['device'])
                    img_lable = img_lable.to(self.FLAGS['System_Parameters']['device']).long()

                    outputs = model(img_vis, img_ir)
                    visualize(image_name=img_name, predictions=outputs.argmax(1))

                    metric_predict = outputs.softmax(dim=1).argmax(dim=1)
                    metric.add_batch(img_lable.detach().cpu().numpy(), metric_predict.detach().cpu().numpy())

                    t.update(1)
                metric.get_testvalue()

    def fuse(self):
        image_path_a, image_path_b, lable_path = testpath()

        # image_path_a, image_path_b, lable_path = test_visualpath()

        self.test_val(image_path_a, image_path_b, lable_path)

