"""
Evaluating inference time (whole HSI) and classification accuracy (test set) of TPPP-Nets
评估TPPP-Nets的推理时间(整体HSI)和分类精度(测试集)
"""
import os
import torch
import argparse
import numpy as np
import yaml
import scipy.io as sio
import time
import auxil
from cls.utils import convert_state_dict
from cls.models import get_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def predict_patches(data, model, cfg, device):
    transfer_data_start = time.time()
    data = data.to(device)
    transfer_data_end = time.time()
    transfer_time = transfer_data_end - transfer_data_start
    predicted = []
    bs = cfg["prediction"]["batch_size"]
    tsp = time.time()
    with torch.no_grad():
        for i in range(0, data.shape[0], bs):
            end_index = i + bs
            batch_data = data[i:end_index]
            outputs = model(batch_data)
            [predicted.append(a) for a in outputs.data.cpu().numpy()]
    tep = time.time()
    prediction_time = tep - tsp
    return prediction_time, transfer_time, np.array(predicted)

from sklearn.decomposition import PCA
def applyPCA(X, numComponents=15):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def timeCost_TPPP(cfg, logdir):
    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    # 设置保存路径
    savepath = './Result/' + datasetname +"_"+ modelname +"_PPsize"+ str(cfg['data']['PPsize'])+"_epochs"+str(cfg['train']['epochs'])+"_PCA"+pca_components+'/'
    try:
        os.makedirs(savepath)
    except FileExistsError:
        pass

    # Setup image
    teposition_path = 'dataset/split_dataset/testSet_position.npy'
    position = np.load(teposition_path)
    org_img_path = 'dataset/'
    if name == "Xiongan":
        data_path = 'dataset/Xiongan/'
        img = sio.loadmat(os.path.join(data_path, 'Xiongan_PCA_15.mat'))['Data']

        #读取ENVI数据
        # import spectral
        # img = spectral.open_image(os.path.join(data_path, 'xiongan_1m_MNFTexture.hdr'))
        # img = img[:, :, :15]
        gt = sio.loadmat(os.path.join(data_path, 'XionganGT.mat'))['data']
        # # PCA or not
        # t = time.gmtime()
        # print(time.strftime("%Y-%m-%d %H:%M:%S", t), "PCAing")
        #
        # num_components = cfg['data']['num_components']
        # if num_components is not None:
        #     img, pca = applyPCA(img, numComponents=num_components)
        #
        # t = time.gmtime()
        # print(time.strftime("%Y-%m-%d %H:%M:%S", t),"PCA is over")
    elif name == "DFC2018":
        data_path = 'dataset/DFC2018/'
        img = sio.loadmat(os.path.join(data_path, 'DFC2018_PCA_15.mat'))['data']
        gt = sio.loadmat(os.path.join(data_path, 'DFC2018_HSI_gt.mat'))['data']
    elif name == "Houston2013":
        data_path = 'dataset/Houston2013/'
        img = sio.loadmat(os.path.join(data_path, 'GRSS2013_HSI_PCA15.mat'))['data']
        gt = sio.loadmat(os.path.join(data_path, 'GRSS2013_Test_GT.mat'))['data']
        # PCA or not
        num_components = cfg['data']['num_components']
        if num_components is not None:
            img, pca = applyPCA(img, numComponents=num_components)

    elif name == "IP":
        img = sio.loadmat(os.path.join(org_img_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        gt = sio.loadmat(os.path.join(org_img_path, 'indian_pines_gt.mat'))['indian_pines_gt']

        # PCA or not
        num_components = cfg['data']['num_components']
        if num_components is not None:
            img, pca = applyPCA(img, numComponents=num_components)

    elif name == "PU":
        img = sio.loadmat(os.path.join(org_img_path, 'paviaU.mat'))['paviaU']
        gt = sio.loadmat(os.path.join(org_img_path, 'paviaU_gt.mat'))['paviaU_gt']

        # PCA or not
        num_components = cfg['data']['num_components']
        if num_components is not None:
            img, pca = applyPCA(img, numComponents=num_components)

    elif name == "SV":
        img = sio.loadmat(os.path.join(org_img_path, 'salinas_corrected.mat'))['salinas_corrected']
        gt = sio.loadmat(os.path.join(org_img_path, 'salinas_gt.mat'))['salinas_gt']

        # PCA or not
        num_components = cfg['data']['num_components']
        if num_components is not None:
            img, pca = applyPCA(img, numComponents=num_components)
    else:
        print("No this dataset")
    print("data shape:", img.shape)
    print("GT shape:", gt.shape)
    # print(min(min(row) for row in gt))
    # print(max(max(row) for row in gt))
    # import sys
    # sys.exit(0)

    # image processing
    time_pre_start = time.time()
    # StandardScaler
    shapeor = img.shape
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler().fit_transform(img)
    img = img.reshape(shapeor)
    # create patch
    data = auxil.creat_PP(cfg["data"]["PPsize"], img)
    # NHWC -> NCHW
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    time_pre_end = time.time()
    time_pre_processing = time_pre_end - time_pre_start
    print("creat patch {} data over!", data.shape)

    # setup model:
    model = get_model(cfg['model'], cfg['data']['dataset'])
    state = convert_state_dict(
        torch.load(os.path.join(logdir, cfg["train"]["best_model_path"]))[
            "model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # transfer model to GPU
    ts1 = time.time()
    model.to(device)
    ts2 = time.time()

    # predicting
    print("predicting...")
    pt, tt, outputs = predict_patches(data, model, cfg, device)
    print(outputs.shape)

    # get result and reshape
    comb_s = time.time()
    outputs = np.array(outputs)
    pred = np.argmax(outputs, axis=1)
    # if cfg['data']['dataset'] == 'IP':
    #     pred = np.reshape(pred, (145, 145))
    # elif cfg['data']['dataset'] == 'Xiongan':
    #     pred = np.reshape(pred, (790, 1875))
    # elif cfg['data']['dataset'] == 'PU':
    #     pred = np.reshape(pred, (610, 340))
    # elif cfg['data']['dataset'] == 'SV':
    #     pred = np.reshape(pred, (512, 217))
    # elif cfg['data']['dataset'] == 'KSC':
    #     pred = np.reshape(pred, (512, 614))
    # elif cfg['data']['dataset'] == 'DFC2018':
    pred = np.reshape(pred, (gt.shape[0], gt.shape[1]))
    comb_e = time.time()

    # show predicted result
    pred += 1
    auxil.decode_segmap(pred)

    # computing classification accuracy
    prednew = pred[position == 1]
    gtnew = gt[position == 1]
    classification, confusion, result = auxil.reports(prednew, gtnew)
    result_info = "OA AA Kappa and each Acc:\n" + str(result)

    # 保存预测结果
    import spectral
    spectral.save_rgb(savepath + datasetname + "_" + modelname + "_predictions_All.jpg", pred.astype(int),
                      colors=spectral.spy_colors)
    mask = np.zeros(gt.shape, dtype='bool')
    mask[gt == 0] = True
    pred[mask] = 0
    spectral.save_rgb(savepath + datasetname +"_"+ modelname+ "_predictions_GT.jpg", pred.astype(int),
                      colors=spectral.spy_colors)

    # report time cost and accuracy
    print("******************** Time ***********************")
    print("Pre_processing time is:", time_pre_processing)
    print("Transfer time is:", tt + (ts2-ts1), "  model:", ts2 - ts1, "  data:", tt)
    print("Prediction time is:", pt)
    print("combine time is:", comb_e-comb_s)
    print('Total inference time is:', time_pre_processing + tt + (ts2-ts1) +pt +comb_e-comb_s)

    # report classification accuracy
    print("****************** Accuracy *********************")
    print(result_info)
    print("****************** classification *********************")
    print(str(classification))
    print("****************** confusion *********************")
    print(str(confusion))

    print("\n")
    file_name = savepath + "classification_report_" + datasetname +"_"+ modelname+"dataset.txt"
    with open(file_name, 'w') as x_file:
        x_file.write("******************** Time ***********************")
        x_file.write('\n')
        x_file.write("Pre_processing time is:{}".format(time_pre_processing))
        x_file.write('\n')
        x_file.write("Transfer time is:{}".format(tt + (ts2 - ts1), "  model:", ts2 - ts1, "  data:", tt))
        x_file.write('\n')
        x_file.write("Prediction time is:{}".format(pt))
        x_file.write('\n')
        x_file.write("combine time is:{}".format(comb_e - comb_s))
        x_file.write('\n')
        x_file.write('Total inference time is:{}'.format(time_pre_processing + tt + (ts2 - ts1) + pt + comb_e - comb_s))
        x_file.write('\n')
        # report classification accuracy
        x_file.write("****************** Accuracy *********************")
        x_file.write('\n')
        x_file.write('{}'.format(result_info))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write("****************** classification *********************")
        x_file.write('\n')
        x_file.write('{}'.format(str(classification)))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write("****************** confusion *********************")
        x_file.write('\n')
        x_file.write('{}'.format(str(confusion)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + '/'+str(cfg["run_ID"])
    timeCost_TPPP(cfg, logdir)
