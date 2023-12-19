import math
import os
import time
import torch
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.Utils import write_yaml



class BaseRunner(object):

    def __init__(self, sys_args):
        super(BaseRunner, self).__init__()
        self.FLAGS = sys_args

    def build_logs(self):
        """
        build log systems
        :return: writer, logs
        """
        # log root path
        times = time.strftime('%Yy%mm%dd-%Hh%Mm', time.localtime())
        log_root_path = self.FLAGS['Root']['checkpoint_root'] + '/' + times

        # build tensorboard
        log_tensorboard = log_root_path + '/tensorboard/'
        if not os.path.exists(log_tensorboard):
            os.makedirs(log_tensorboard)
        writer = SummaryWriter(log_tensorboard)

        # build run_tensorboard.bat
        run_tensorboard_path = log_root_path + '/run_tensorboard.bat'
        f = open(run_tensorboard_path, 'a')
        f.write('cmd/k "conda activate zzw.torch1.12.1&&cd/d ' + os.path.abspath('.\Checkpoint')
                + '\\' + times + '\\&&' + 'tensorboard --logdir=tensorboard --port=6006"')
        f.close()

        # build train_logs with .yaml file
        log_yaml = self.FLAGS['Root']['checkpoint_root'] + '/' + times + '/checkpoint/'

        if not os.path.exists(log_yaml):
            os.makedirs(log_yaml)
        self.FLAGS['Root']['log_path'] = log_yaml

        # build demo save path
        demo_save_path = self.FLAGS['Root']['checkpoint_root'] + '/' + times + '/demo_save/'

        if not os.path.exists(demo_save_path):
            os.makedirs(demo_save_path)
        self.FLAGS['Root']['demo_save_path'] = demo_save_path

        LOGS = {'Times': str(times), 'Train': {}, 'Val': {}, 'Root': {}, 'Network': {}, 'Params': {}}
        write_yaml(log=LOGS, log_path=log_yaml)
        return writer, LOGS

    def build_model(self, LOGS):

        from Convnextv2.builder import Convnextv2
        # MS-IRTNet
        netG= Convnextv2(in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(self.FLAGS['System_Parameters']['device'])

        print('-' * 35)
        print('---Build NetWork Done!')
        from Utils.Utils import get_parameter_number
        number_of_parameters = get_parameter_number(netG).get('Total')
        LOGS['Network']['number_of_parameters'] = number_of_parameters
        write_yaml(log=LOGS, log_path=self.FLAGS['Root']['log_path'])
        print('   MS-IRTNet_Parameters:', number_of_parameters)
        return netG

    def load_state(self, net):

        if self.FLAGS['System_Parameters']['use_pretrained_model'] == 'True' or self.FLAGS['is_test'] == 'True':

            model_path = self.FLAGS['Root']['checkpoint_root'] + '/' + self.FLAGS['Test']['checkpoint_name'] \
                         + '/checkpoint/' + 'Model_epoch_' + self.FLAGS['Test']['model_epoch'] + '.pth'

            if os.path.exists(model_path):
                print('   model_path:', model_path)
                resume_epoch = torch.load(model_path)['epoch']

                net.load_state_dict(
                    torch.load(model_path,
                               map_location=lambda storage, location: storage)['state_dict']
                )

                print('---Load NetWork Weights Done!')
                return net, resume_epoch
            else:
                raise Exception('No pretrained model loaded!!!\nPlease check path: "./Checkpoint/(time)/Models.."')
        else:
            raise Exception('Both use_pretrained_model & is_test is False!!! check usage')

    def build_datasets(self, datatype):
        """
        load Data.
        :return: data, loader
        """

        from Data.Dataloader_voxel import MotherData

        VAL_DATASET = {}

        if datatype == 'train':
            self.FLAGS['Data']['data_type'] = 'train'
            train_data = MotherData(
                directory=self.FLAGS['Root']['data_dictionary_train'],
                FLAGS=self.FLAGS,
            )

            train_loader = DataLoader(
                train_data,
                batch_size=self.FLAGS['Train']['train_batch_size'],
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
                num_workers=self.FLAGS['System_Parameters']['num_workers']
            )

            train_data_num = len(train_data)
            print('-' * 35)
            print('---Load Train_data Done!')
            print('   Train_data_num: %s' % len(train_data))
            TRAIN_DATASET = {'loader': train_loader, 'data_num': train_data_num}
            return TRAIN_DATASET

        if datatype == 'valid':

            for val_dataset in self.FLAGS['Data']['split_sequences']['valid']:
                self.FLAGS['Data']['data_type'] = val_dataset
                VAL_DATASET[val_dataset] = {}
                valid_data = MotherData(
                    directory=self.FLAGS['Root']['data_dictionary_val'],
                    FLAGS=self.FLAGS,
                )

                valid_loader = DataLoader(
                    valid_data,
                    batch_size=self.FLAGS['Train']['val_batch_size'],
                    shuffle=False,
                    pin_memory=False,
                    num_workers=self.FLAGS['System_Parameters']['num_workers']
                )

                valid_data_num = len(valid_data)

                print('---Load Val_data Done!')
                print('   ' + val_dataset + '_data_num: %s' % len(valid_data))
                VAL_DATASET[val_dataset]['loader'] = valid_loader
                VAL_DATASET[val_dataset]['data_num'] = valid_data_num
            return VAL_DATASET

    def import_loss_functions(self):
        """
        build loss_functions

        :return: loss
        """

        print('-' * 35)
        LOSS = {
            'CEN': torch.nn.CrossEntropyLoss(ignore_index=255)
        }
        print('---Build Loss Done!')
        return LOSS

    def setup_optimizer(self, netG, LOGS):
        """
        setup optimizer, include lr, weight_decay, scheduler
        :return: optimizerG, lr_policy
        """

        from Utils.lr_policy import WarmUpPolyLR
        optimizerG = torch.optim.AdamW(netG.parameters(),
                                       lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
        lr_policy = WarmUpPolyLR(
            6e-5,
            0.9,
            784 * self.FLAGS['Train']['epoch'],
            (784 // self.FLAGS['Train']['val_batch_size'] + 1) * 10)

        print('-' * 35)
        print('---Build Optimizer Done!')
        print('   G_learning_rate:%s\n   decay_step:%s\n   decay_rate:%s' %
              (self.FLAGS['Train']['learning_rate'], self.FLAGS['Train']['decay_step'],
               self.FLAGS['Train']['decay_rate']))

        LOGS['Params']['learning_rate'] = self.FLAGS['Train']['learning_rate']
        LOGS['Params']['decay_rate'] = self.FLAGS['Train']['decay_rate']
        LOGS['Params']['decay_step'] = self.FLAGS['Train']['decay_step']
        write_yaml(log=LOGS, log_path=self.FLAGS['Root']['log_path'])

        return optimizerG, lr_policy

    def save_bestmodel(self, model_need_save):

        torch.save(
            {
             'state_dict': model_need_save.state_dict()},
            self.FLAGS['Root']['log_path'] + '/Model' + '_statedict' + '.pth'
        )
        torch.save(model_need_save, self.FLAGS['Root']['log_path'] + 'model.pth')


    def build_metrics(self):

        from Utils.metrics import SegEvaluator

        Metrics = SegEvaluator(num_class=9, FLAGS=self.FLAGS)

        return Metrics


class Train(BaseRunner):

    def val_while_train(self, val_loader, valid_data_num, net, epoch, metrics, writer, valid_dataset, LOGS):

        self.FLAGS['Data']['data_type'] = 'valid'

        with torch.no_grad():

            net.eval()
            metrics.reset()
            if epoch % self.FLAGS['Train']['epoch'] == 0:
                if not os.path.exists(self.FLAGS['Root']['demo_save_path'] + str(valid_dataset) + '/'):
                    os.makedirs(self.FLAGS['Root']['demo_save_path'] + str(valid_dataset) + '/')

            all_time = 0

            with tqdm(total=valid_data_num, ncols=100, ascii=True) as t:
                for i, VALID_DATA in enumerate(val_loader, 0):
                    t.set_description(valid_dataset + 'Val_num %s' % i)

                    val_ir = VALID_DATA['image_IR'].to(self.FLAGS['System_Parameters']['device'])
                    val_vi = VALID_DATA['image_VI'].to(self.FLAGS['System_Parameters']['device'])

                    val_label = VALID_DATA['image_LABEL'].to(self.FLAGS['System_Parameters']['device']).long()

                    start_time = time.time()
                    output = net(val_vi, val_ir)

                    predict = output.softmax(dim=1).argmax(dim=1)
                    per_time = time.time() - start_time
                    all_time += per_time

                    metrics.add_batch(val_label.detach().cpu().numpy(), predict.detach().cpu().numpy())

                    if i % 100 == 0:
                        writer.add_images('Test-{}/IR'.format(valid_dataset), val_ir, epoch)
                        writer.add_images('Test-{}/Vi'.format(valid_dataset), val_vi, epoch)
                        writer.add_images('Test-{}/PRE'.format(valid_dataset), predict.detach().unsqueeze(1),
                                          epoch)
                        writer.add_images('Test-{}/LABEL'.format(valid_dataset), val_label.detach().unsqueeze(1), epoch)

                    t.update(self.FLAGS['Train']['val_batch_size'])

                LOGS['Network'][valid_dataset]['inference_time'] = all_time
                # metrics.get_value(epoch=epoch, writer=writer, LOGS=LOGS, valid_dataset=valid_dataset)
                miou = metrics.get_value(epoch=epoch, writer=writer, LOGS=LOGS, valid_dataset=valid_dataset)

                return miou

    def train(self, train_loader, train_data_num, epoch, writer, netG, optimizerG, schedulerG,
              LOSS, LOGS):

        self.FLAGS['Data']['data_type'] = 'train'
        cen_loss = LOSS['CEN']

        netG.train()

        with tqdm(total=train_data_num, ncols=100, ascii=True) as t:

            for i, TRAIN_DATA in enumerate(train_loader, 0):
                t.set_description('Epoch:[%s/%s]-Iteration:[%s/%s]' %
                                  (epoch, self.FLAGS['Train']['epoch'],
                                   (epoch - 1) * train_data_num + i * self.FLAGS['Train']['train_batch_size'] + 1,
                                   train_data_num * self.FLAGS['Train']['epoch']))

                train_ir = TRAIN_DATA['image_IR'].to(self.FLAGS['System_Parameters']['device'])
                train_vi = TRAIN_DATA['image_VI'].to(self.FLAGS['System_Parameters']['device'])

                train_label = TRAIN_DATA['image_LABEL'].to(self.FLAGS['System_Parameters']['device']).long()

                train_output = netG(train_vi, train_ir)

                total_loss_value = cen_loss(train_output, train_label)

                optimizerG.zero_grad()

                total_loss_value.backward()
                optimizerG.step()

                if i % 50 == 0:
                    writer.add_images('Train/IR', train_ir, epoch)
                    writer.add_images('Train/Vi', train_vi, epoch)
                    writer.add_images('Train/PRE', train_output.detach().softmax(dim=1).argmax(dim=1).unsqueeze(1),
                                      epoch)
                    writer.add_images('Train/LABEL', train_label.detach().unsqueeze(1), epoch)

                # tensorboard log for loss
                writer.add_scalar('Train/SEG-Loss', total_loss_value.detach(),
                                  (epoch - 1) * train_data_num + i * self.FLAGS['Train']['train_batch_size'] + 1)

                t.update(self.FLAGS['Train']['train_batch_size'])


                current_idx = (epoch - 1) * (784 // self.FLAGS['Train']['val_batch_size'] + 1) + i
                lr = schedulerG.get_lr(current_idx)

                writer.add_scalar('Train/Lr', lr,
                                  (epoch - 1) * train_data_num + i * self.FLAGS['Train']['val_batch_size'] + 1)

                for i in range(len(optimizerG.param_groups)):
                    optimizerG.param_groups[i]['lr'] = lr

    def run_train(self):
        """
        training process
        """
        writer, LOGS = self.build_logs()

        TRAIN_DATASET = self.build_datasets(datatype='train')
        VAL_DATASET = self.build_datasets(datatype='valid')

        metrics = self.build_metrics()
        LOSS = self.import_loss_functions()

        netG = self.build_model(LOGS=LOGS)

        if self.FLAGS['System_Parameters']['use_pretrained_model'] == 'True':
            netG, resume_epoch = self.load_state(net=netG)
            resume_epoch = self.FLAGS['Test']['start_epoch']

        else:
            resume_epoch = 1
        optimizerG, schedulerG = self.setup_optimizer(netG=netG, LOGS=LOGS)

        print('-' * 35)
        print('---Start Training')
        time.sleep(0.5)
        current = 0
        for epoch in range(resume_epoch, self.FLAGS['Train']['epoch'] + 1):

            if self.FLAGS['System_Parameters']['test'] == 'False':
                self.train(
                    train_loader=TRAIN_DATASET['loader'],
                    train_data_num=TRAIN_DATASET['data_num'],
                    epoch=epoch,
                    writer=writer,
                    netG=netG,
                    optimizerG=optimizerG,
                    schedulerG=schedulerG,
                    LOSS=LOSS,
                    LOGS=LOGS
                )

            if self.FLAGS['System_Parameters']['Val_while_train'] == 'True':

                if epoch % 1 == 0:

                    LOGS['Val']['epoch-%03d' % epoch] = {}

                    for valid_dataset in self.FLAGS['Data']['split_sequences']['valid']:
                        LOGS['Network'][valid_dataset] = {}

                        time.sleep(0.5)

                        miou = self.val_while_train(
                            val_loader=VAL_DATASET[valid_dataset]['loader'],
                            valid_data_num=VAL_DATASET[valid_dataset]['data_num'],
                            epoch=epoch,
                            writer=writer,
                            net=netG,
                            metrics=metrics,
                            valid_dataset=valid_dataset,
                            LOGS=LOGS
                        )
                        if(miou > current):
                            print("epoch:", epoch, "miou=", miou)
                            print("current_miou", current)
                            current = miou
                            self.save_bestmodel(netG)


