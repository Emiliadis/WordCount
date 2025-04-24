import os
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

from model import embed_net
from utils import *


# ----------------------------
# 封装模型加载函数
# ----------------------------
def load_model(model_path, resume, dataset='dn348', gpu='0', img_w=256, img_h=256):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # 根据数据集设定类别数和测试模式
    if dataset.lower() == 'dn348':
        n_class = 200
        test_mode = [2, 2]  # 比如：gallery 使用 test_mode[0]，query 使用 test_mode[1]
    elif dataset.lower() == 'dnwild':
        n_class = 1574
        test_mode = [2, 1]
    else:
        raise ValueError("仅支持 dn348 或 dnwild 数据集！")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')
    net = embed_net(n_class, no_local='on', gm_pool='on', arch='resnet50')
    net.to(device)
    cudnn.benchmark = True

    checkpoint_file = os.path.join(model_path, resume)
    print('==> Resuming from checkpoint: {}'.format(checkpoint_file))
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net.load_state_dict(checkpoint['net'])
        print('==> Loaded checkpoint {} (epoch {})'.format(resume, checkpoint.get('epoch', 'N/A')))
    else:
        raise FileNotFoundError('==> No checkpoint found at {}'.format(checkpoint_file))

    # 定义预处理，注意 Resize 的尺寸：Resize 接受 (h, w)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    return net, test_mode, device, transform_test, n_class


# ----------------------------
# 自定义数据集
# ----------------------------
class CustomDataset(data.Dataset):
    def __init__(self, img_dir, transform=None, img_size=(256, 256)):
        self.img_dir = img_dir
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert('RGB')
        img = img.resize(self.img_size, Image.ANTIALIAS)
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        # 返回图片张量及对应路径
        return img_tensor, path

    def __len__(self):
        return len(self.img_paths)


# ----------------------------
# 提取单张图像特征函数（用于查询图片）
# ----------------------------
def extract_features_image(img_tensor, net, device, test_mode_value):
    net.eval()
    with torch.no_grad():
        # 单张图片需要增加batch维度
        img_tensor = img_tensor.unsqueeze(0).to(device)
        # 对于查询图片，这里按照模型接口传入四次同样的输入，
        # 并使用 test_mode_value（通常对应 test_mode[1]）
        _, feat_fc = net(img_tensor, img_tensor, img_tensor, img_tensor, test_mode_value)
        feat = feat_fc.detach().cpu().numpy()
    return feat


# ----------------------------
# 提取整个 gallery 数据集特征
# ----------------------------
def extract_features_loader(loader, net, device, test_mode_value):
    net.eval()
    feats = []
    paths_all = []
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            imgs, paths = batch
            imgs = imgs.to(device)
            # 对 gallery 图片，使用 test_mode[0]
            _, feat_fc = net(imgs, imgs, imgs, imgs, test_mode_value)
            feats.append(feat_fc.detach().cpu().numpy())
            paths_all += list(paths)
    feats = np.vstack(feats)
    print('Feature extraction done in {:.3f} seconds.'.format(time.time() - start_time))
    return feats, paths_all


# ----------------------------
# 对外接口函数：重识别
# ----------------------------
def re_identify(query_image_path, model_params, gallery_dir, topk=3):
    """
    参数：
        query_image_path - 上传的查询图片路径
        model_params - load_model 返回的元组 (net, test_mode, device, transform_test, n_class)
        gallery_dir - gallery 文件夹的路径
        topk - 返回相似度最高的 topk 个结果（默认 3）
    返回：
        列表，每个元素为 (gallery_image_path, similarity_score)
    """
    net, test_mode, device, transform_test, n_class = model_params

    # 加载查询图片，并预处理
    query_img = Image.open(query_image_path).convert('RGB')
    # 注意：这里的 resize 尺寸需与 transform_test 中保持一致 (img_h, img_w)
    query_img = query_img.resize(transform_test.transforms[0].size, Image.ANTIALIAS)
    query_tensor = transform_test(query_img)
    # 提取查询图片特征，使用 test_mode[1]（假设 query 用该模式）
    query_feat = extract_features_image(query_tensor, net, device, test_mode_value=test_mode[1])

    # 构建 gallery 数据集
    # gallery 中图片调整到与 transform_test 保持一致的尺寸
    # 注意：transform_test.transforms[0].size 可能需转换为 (w, h) 格式
    img_size = (transform_test.transforms[0].size[1], transform_test.transforms[0].size[0])
    gallery_dataset = CustomDataset(gallery_dir, transform=transform_test, img_size=img_size)
    gallery_loader = data.DataLoader(gallery_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 提取 gallery 图片特征，使用 test_mode[0]
    gallery_feats, gallery_paths = extract_features_loader(gallery_loader, net, device, test_mode_value=test_mode[0])

    # 计算查询图片与 gallery 图片间的相似度
    # 使用内积作为相似度分数（内积越大表示相似度越高）
    distmat = np.matmul(query_feat, gallery_feats.T)  # 形状为 (1, num_gallery)

    # 获取 topk 个结果，返回时将相似度和对应的图片路径一起返回
    indices = np.argsort(-distmat, axis=1)[0][:topk]
    results = []
    for idx in indices:
        results.append((gallery_paths[idx], float(distmat[0, idx])))

    return results


# ----------------------------
# 以下为命令行运行的入口，仅供调试，不影响 Vehicle_D.py 导入使用
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Re-ID Test for Custom Query/Gallery Data')
    parser.add_argument('--dataset', default='dn348', help='仅支持 dn348 或 dnwild')
    parser.add_argument('--mode', default='all', type=str, help='运行模式 (仅留 all 模式)')
    parser.add_argument('--resume', '-r', required=True, type=str, help='模型文件名称（位于 model_path 下）')
    parser.add_argument('--model_path', default='save_model/', type=str, help='模型保存路径')
    parser.add_argument('--gpu', default='0', type=str, help='GPU设备ID')
    parser.add_argument('--img_w', default=256, type=int, help='图像宽度')
    parser.add_argument('--img_h', default=256, type=int, help='图像高度')
    parser.add_argument('--gallery_dir', default='/media/emilia/新加卷/DN ReID/DNDM/gallery', type=str,
                        help='gallery 文件夹路径')
    parser.add_argument('--query_image', default='path_to_query_image.jpg', type=str, help='查询图片路径')
    args = parser.parse_args()

    model_params = load_model(args.model_path, args.resume, dataset=args.dataset, gpu=args.gpu, img_w=args.img_w,
                              img_h=args.img_h)
    results = re_identify(args.query_image, model_params, args.gallery_dir, topk=3)
    print("Top 3 matches:")
    for res in results:
        print(res)
