import os,sys
import torch
import torch.nn as nn
import torch.optim as optim

from bop_toolkit_lib import inout
from pix2pose_model_torch import torch_models
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import GeneratorEnqueuer
from pix2pose_util import data_io as dataio
from tools import bop_io

if(len(sys.argv)!=6):
    print("python3 tools/3_train_pix2pose.py <gpu_id> <cfg_fn> <dataset> <obj_id> <dir_to_background_imgs>")
    sys.exit()

gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#using cpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")


def dummy_loss(y_true,y_pred):
    return y_pred

##1: 这个func是：若输入是fake-img，则使用Generator进行生成，并声成其对应的label0；若输入是real-image，则直接生成label1即可
def get_disc_batch(X_src, X_tgt, generator_model, batch_counter,label_smoothing=False,label_flipping=0):    
    if batch_counter % 2 == 0:
        ##generator_model.weights.double()
        X_disc,prob_dummy = generator_model(X_src)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        if label_smoothing:
            y_disc = np.random.uniform(low=0.0, high=0.1, size=y_disc.shape[0])            
        else:
            y_disc = 0
            
        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:] = 1
    else:
        X_disc = X_tgt
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc = np.random.uniform(low=0.9, high=1.0, size=y_disc.shape[0])                
        else:
            y_disc = 1                
        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:] = 0

    return X_disc, y_disc


loss_weights = [100,1]
#train_gen_first = False
#load_recent_weight = True


dataset=sys.argv[3]

cfg_fn = sys.argv[2] #"cfg/cfg_bop2019.json"
cfg = inout.load_json(cfg_fn)

##2： 这里是配备一堆路径
bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True)
im_width,im_height =cam_param_global['im_size'] 
weight_prefix = "pix2pose" 
obj_id = int(sys.argv[4]) #identical to the number for the ply file.
weight_dir = bop_dir+"/pix2pose_weights/{:02d}".format(obj_id)
if not(os.path.exists(weight_dir)):
        os.makedirs(weight_dir)
back_dir = sys.argv[5]
data_dir = bop_dir+"/train_xyz/{:02d}".format(obj_id)

batch_size=50
datagenerator = dataio.data_generator(data_dir,back_dir,batch_size=batch_size,res_x=im_width,res_y=im_height)

m_info = model_info['{}'.format(obj_id)]
keys = m_info.keys()
sym_pool=[]
sym_cont = False
sym_pool.append(np.eye(3))
if('symmetries_discrete' in keys):
    print(obj_id,"is symmetric_discrete")
    print("During the training, discrete transform will be properly handled by transformer loss")
    sym_poses = m_info['symmetries_discrete']
    print("List of the symmetric pose(s)")
    for sym_pose in sym_poses:
        sym_pose = np.array(sym_pose).reshape(4,4)
        print(sym_pose[:3,:3])
        sym_pool.append(sym_pose[:3,:3])
if('symmetries_continuous' in keys):
    sym_cont=True

#optimizer_dcgan =Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#optimizer_disc = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

##3: 这里是加载model
backbone='paper'
if('backbone' in cfg.keys()):
    if(cfg['backbone']=="resnet50"):
            backbone='resnet50'
#if(backbone=='resnet50'):
#    generator_train = ae.aemodel_unet_resnet50(p=1.0)
#else:
#    generator_train = ae.aemodel_unet_prob(p=1.0)
generator_train = torch_models.Unet().double().cuda()
discriminator = torch_models.discriminator().double().cuda()
#gen_img, prob = generator_train()

epoch=0
recent_epoch=-1

##4： 这里是加载记录的中间model，这里不知道怎么写成torch版本？？？
#if load_recent_weight:
#    weight_save_gen=""
#    weight_save_disc=""
#    for fn_temp in sorted(os.listdir(weight_dir)):
#        if(fn_temp.startswith(weight_prefix+".")):
#                    temp_split  = fn_temp.split(".")
#                    epoch_split = temp_split[1].split("-") #"01_real_1.0-0.1752.hdf5"
#                    epoch_split2= epoch_split[0].split("_") #01_real_1.0
#                    epoch_temp = int(epoch_split2[0])
#                    network_part = epoch_split2[1]
#                    if(epoch_temp>=recent_epoch):
#                        recent_epoch = epoch_temp
#                        if(network_part=="gen"):
#                            weight_save_gen = fn_temp
#                        elif(network_part=="disc"):
#                            weight_save_disc = fn_temp
#

#    if(weight_save_gen!=""):
#        print("load recent weights from", weight_dir+"/"+weight_save_gen)
#        generator_train.load_state_dict(os.path.join(weight_dir,weight_save_gen))
#        ##generator_train.load_weights(os.path.join(weight_dir,weight_save_gen))
    
#    if(weight_save_disc!=""):
#        print("load recent weights from", weight_dir+"/"+weight_save_disc)
#        discriminator.load_state_dict(os.path.join(weight_dir,weight_save_disc))
#        ##discriminator.load_weights(os.path.join(weight_dir,weight_save_disc))
   
##5： lr会发生修改
if(recent_epoch!=-1):
    epoch = recent_epoch
    train_gen_first=False
max_epoch=10
if(max_epoch==10): #lr-shcedule used in the bop challenge
    lr_schedule=[1E-3,1E-3,1E-3,1E-3,1E-3,
                1E-3,1E-3,1E-4,1E-4,1E-4,
                1E-5,1E-5,1E-5,1E-5,1E-6,
                1E-6,1E-6,1E-6,1E-6,1E-7]
elif(max_epoch==20): #lr-shcedule used in the paper
    lr_schedule=[1E-3,1E-3,1E-3,1E-3,1E-3,
                1E-3,1E-3,1E-3,1E-3,1E-4,
                1E-4,1E-4,1E-4,1E-4,1E-4,
                1E-4,1E-4,1E-4,1E-4,1E-5]

##6： 这里是定义基本参数
criterion = nn.BCELoss().to(device)
#optimizer_G = optim.Adam(generator_train.parameters(), lr=1E-4, betas=(0.9, 0.999), eps=1e-08)
#optimizer_D = optim.Adam(discriminator.parameters(), lr=1E-4, betas=(0.9, 0.999), eps=1e-08)

N_data=datagenerator.n_data
batch_size= 50
batch_counter=0
n_batch_per_epoch= min(N_data/batch_size*10,3000) #check point: every 10 epoch
step_lr_drop=5

disc_losses=[]
recont_losses=[]
gen_losses=[]
pre_loss=9999
lr_current=lr_schedule[epoch]

real_ratio=1.0
feed_iter= datagenerator.generator()
print(feed_iter)
optimizer_G = optim.Adam(generator_train.parameters(), lr=lr_current, betas=(0.9, 0.999), eps=1e-08)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_current, betas=(0.9, 0.999), eps=1e-08)

##7： 这是一个tensorflow的加载器，不知道怎么改成torch版本？？？？？
fed = GeneratorEnqueuer(feed_iter,use_multiprocessing=False, wait_time=5) #####1: ?????
fed.start(workers=1,max_queue_size=200)
iter_ = fed.get()

zero_target = torch.zeros((batch_size))

##8： 开始按照batch进行训练
for X_src,X_tgt,disc_tgt,prob_gt in iter_:
    optimizer_D.zero_grad()
    #1: change data to torch.tensor
    X_src = torch.tensor(X_src).permute(0, 3, 1, 2).double().cuda()
    X_tgt = torch.tensor(X_tgt).permute(0, 3, 1, 2).double().cuda()
    prob_gt = torch.tensor(prob_gt).cuda()

    ##9： Update D
    X_disc, y_disc = get_disc_batch(X_src,X_tgt,generator_train,0,
                                    label_smoothing=True,label_flipping=0.2)
    X_disc, y_disc = torch.tensor(X_disc).to(device), torch.tensor(y_disc).to(device)
    y_disc = torch.tensor(y_disc).view(-1, 1).to(device)
    disc_loss = discriminator(X_disc, X_tgt)

    disc_loss = criterion(disc_loss, y_disc)

    X_disc, y_disc = get_disc_batch(X_src,X_tgt,generator_train,1,
                                    label_smoothing=True,label_flipping=0.2)
    X_disc, y_disc = torch.tensor(X_disc).to(device), torch.tensor(y_disc).to(device)
    y_disc = torch.tensor(y_disc).view(-1, 1).to(device)
    disc_loss2 = discriminator(X_disc, X_tgt)
    disc_loss2 = criterion(disc_loss2, y_disc)
    disc_loss = (disc_loss + disc_loss2)/2
    disc_loss.backward()
    optimizer_D.step()

    disc_losses.append(disc_loss)

    ##10： Update G
    optimizer_G.zero_grad()
    gen_img, prob = generator_train(X_src)
    recont_l = torch_models.transformer_loss(gen_img, X_tgt, prob, prob_gt, sym_pool)#y_pred, y_recont_gt, y_prob_pred, y_prob_gt, sym
    disc_out = discriminator(gen_img, X_tgt)
    dcgan_loss = criterion(disc_out, torch.ones_like((disc_out)))
    dcgan_loss.backward()
    optimizer_D.step()

    recont_losses.append(recont_l)
    gen_losses.append(dcgan_loss)

    ##11： 这里不知道怎么进行记录的修改？？？
    #mean_loss = np.mean(np.array(recont_losses))
    ##print("Epoch{:02d}-Iter{:03d}/{:03d}:Mean-[{:.5f}], Disc-[{:.4f}], Recon-[{:.4f}], Gen-[{:.4f}]],lr={:.6f}".format(epoch,batch_counter,int(n_batch_per_epoch),mean_loss,disc_loss.detach().numpy(),recont_l.detach().numpy(),dcgan_loss.detach().numpy(),lr_current))
    print("running")
    if(batch_counter>n_batch_per_epoch):
        #mean_loss = np.mean(np.array(recont_losses))
        disc_losses=[]
        recont_losses=[]
        gen_losses=[]
        batch_counter=0
        epoch+=1
        print('disc_loss:',disc_loss)
        print('dcgan_loss:',dcgan_loss)
        ##12： 这里不知道怎么正确保存数据？？？
        #if( mean_loss< pre_loss):
        #    print("loss improved from {:.4f} to {:.4f} saved weights".format(pre_loss,mean_loss))
        #    print(weight_dir+"/"+weight_prefix+".{:02d}-{:.4f}.hdf5".format(epoch,mean_loss))
        #    pre_loss=mean_loss
        #else:
        #    print("loss was not improved")
        #    print(weight_dir+"/"+weight_prefix+".{:02d}-{:.4f}.hdf5".format(epoch,mean_loss))

        #weight_save_gen = weight_dir+"/" + weight_prefix+".{:02d}_gen_{:.1f}-{:.4f}.hdf5".format(epoch,real_ratio,mean_loss)
        #weight_save_disc = weight_dir+"/" + weight_prefix+".{:02d}_disc_{:.1f}-{:.4f}.hdf5".format(epoch,real_ratio,mean_loss)
        #generator_train.save_weights(weight_save_gen)
        #discriminator.save_weights(weight_save_disc)
        
        gen_images,probs = generator_train.predict(X_src)

        imgfn = weight_dir+"/val_img/"+weight_prefix+"_{:02d}.png".format(epoch)
        if not(os.path.exists(weight_dir+"/val_img/")):
            os.makedirs(weight_dir+"/val_img/")
        
        f,ax=plt.subplots(10,3,figsize=(10,20))
        for i in range(10):
            ax[i,0].imshow( (X_src[i]+1)/2)
            ax[i,1].imshow( (X_tgt[i]+1)/2)
            ax[i,2].imshow( (gen_images[i]+1)/2)
        plt.savefig(imgfn)
        plt.close()
        
        lr_current=lr_schedule[epoch]
        optimizer_G = optim.Adam(generator_train.parameters(), lr=lr_current, betas=(0.9, 0.999), eps=1e-08)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_current, betas=(0.9, 0.999), eps=1e-08)

    batch_counter+=1
    ##13： 这里不知道该怎么保存文件才对？？？
    #if(epoch>max_epoch):
    #    print("Train finished")
    #    if(backbone=='paper'):
    #        generator_train.save_weights(os.path.join(weight_dir,"inference.hdf5"))
    #    else:
    #        generator_train.save(os.path.join(weight_dir,"inference_resnet_model.hdf5"))
    #    break
