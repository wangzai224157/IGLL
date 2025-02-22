import numpy as np
import cv2
import os
import subprocess
import glob
from options.test_options import TestOptions
from model.net import InpaintingModel_DFBM
from util.utils import generate_rect_mask, generate_stroke_mask, getLatest

config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
    pathfile = [os.path.join(r'/home1/yzhou/data/CelebA/Img/img_align_celeba_png', x) for x in pathfile]

elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.jpg'))
    pathfile_edge = glob.glob(os.path.join(config.edge_dir, '*.jpg'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)

total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

print('configuring model..')
ourModel = InpaintingModel_DFBM(opt=config)
ourModel.print_networks()

"""
dataset = InpaintingDataset(config.data_file,config.dataset_path , config.edge_dir, transform=transforms.Compose([
    ToTensor() ]))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)
for i, data in enumerate(dataloader):
    gt = data['gt'].cuda()
    # normalize to values between -1 and 1
    gt = gt / 127.5 - 1
    edge = data['edge'].cuda()
    data_in = {'gt': gt}
    ourModel.setInput(data_in, edge)

"""


if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')




if config.random_mask:
    np.random.seed(config.seed)

for i in range(test_num):
    print("test")
    if config.mask_type == 'rect':
        mask, _ = generate_rect_mask(config.img_shapes, config.mask_shapes, rand_mask=False)
    else:
        mask = generate_stroke_mask(im_size=(config.img_shapes[0], config.img_shapes[1]),
                                    parts=8, maxBrushWidth=20, maxLength=100, maxVertex=20)

    image = cv2.imread(pathfile[i])
    edge = cv2.imread(pathfile_edge[i])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    """
    if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
        h_start = (h - config.img_shapes[0]) // 2
        w_start = (w - config.img_shapes[1]) // 2
        image = image[h_start: h_start + config.img_shapes[0], w_start: w_start + config.img_shapes[1], :]
        edge = edge[h_start: h_start + config.img_shapes[0], w_start: w_start + config.img_shapes[1], :]
    else:
        t = min(h, w)
        image = image[(h - t) // 2:(h - t) // 2 + t, (w - t) // 2:(w - t) // 2 + t, :]
        image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))
        edge = edge[(h - t) // 2:(h - t) // 2 + t, (w - t) // 2:(w - t) // 2 + t, :]
        edge = cv2.resize(edge, (config.img_shapes[1], config.img_shapes[0]))
    """
    image = image[50: 50 + config.img_shapes[0], 50: 50 + config.img_shapes[1], :]
    edge = edge[50: 50 + config.img_shapes[0], 50: 50 + config.img_shapes[1], :]



    image = np.transpose(image, [2, 0, 1]) 
    image = np.expand_dims(image, axis=0)
    image_clear = np.transpose(image[0][::-1, :, :], [1, 2, 0])
    cv2.imwrite(os.path.join(config.clean_path, '{:03d}.jpg'.format(i)), image_clear.astype(np.uint8))

    edge = np.transpose(edge, [2, 0, 1])
    edge = np.expand_dims(edge, axis=0) 
   

    image_vis = image * (1 - mask) + 255 * mask
    image_vis = np.transpose(image_vis[0][::-1, :, :], [1, 2, 0])
    edge_vis = np.transpose(edge[0][::-1, :, :], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, 'input_{:03d}.jpg'.format(i)), image_vis.astype(np.uint8))
    cv2.imwrite(os.path.join(config.saving_path, 'input_{:03d}.jpg'.format(i)), edge_vis.astype(np.uint8))



    h, w = image.shape[2:]
    grid = 4
    image = image[:, :, :h // grid * grid, :w // grid * grid]
    mask = mask[:, :, :h // grid * grid, :w // grid * grid]
    edge = edge[:, :, :h // grid * grid, :w // grid * grid]
    #input_data = np.append(image,edge,axis=1)
    result = ourModel.evaluate(image,edge, mask)
    
    result = np.transpose(result[0][::-1, :, :], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '{:03d}.jpg'.format(i)), result)
    print(' > {} / {}'.format(i + 1, test_num)) 

print('done.')
