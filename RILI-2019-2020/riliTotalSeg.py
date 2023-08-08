import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from model import UNet
from torchvision import transforms as xfms
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.python_api import totalsegmentator

HU_CLIP = (-900, 124)
IMG_SIZE = 512
NET_SIZE = 448
MEAN = 0.36
STD = 0.42
BATCH_SIZE = 8
XFM_COMP = xfms.Compose([
    xfms.ToTensor(),
    xfms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),  # type: ignore
    xfms.CenterCrop((NET_SIZE, NET_SIZE)),
    xfms.Normalize(MEAN, STD)
])

GPU_NUMBER = 0
THRESHOLD = 0.5
SMOOTH_AREA = 150

NET = UNet(outSize=(NET_SIZE, NET_SIZE))


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def toITK(pixels: np.ndarray, savePath: Path, affine: dict):
    # pixels: 3D numpy array
    # path: save path
    # infos: dict
    itk = sitk.GetImageFromArray(pixels)
    itk.SetSpacing(affine['spacing'])
    itk.SetOrigin(affine['origin'])
    itk.SetDirection(affine['direction'])

    sitk.WriteImage(itk, str(savePath))
    return itk


def info(*args, show: bool = True, **kwargs):
    if show:
        print(*args, **kwargs)


def seg_simplify(seg_itk: sitk.Image) -> np.ndarray:
    seg_ary = sitk.GetArrayFromImage(seg_itk)
    seg_simp_ary = np.zeros_like(seg_ary, dtype=np.uint8)
    '''
             0,      1,    2-3,     5,    6,    13-17, 44-48,    18-41, 58-81, X
    background, spleen, kidney, liver, stomach, lung, heart, vertebrae, rib, rili
             0,      1,      2,     3,       4,    5,     6,         7,   8,    9
    '''
    seg_simp_ary[seg_ary == 1] = 1  # seg spleen
    seg_simp_ary[(seg_ary == 2) | (seg_ary == 3)] = 2  # seg kidney
    seg_simp_ary[seg_ary == 5] = 3  # seg liver
    seg_simp_ary[seg_ary == 6] = 4  # seg stomach
    seg_simp_ary[(seg_ary >= 13) & (seg_ary <= 17)] = 5  # seg lung
    seg_simp_ary[(seg_ary >= 44) & (seg_ary <= 48)] = 6  # seg heart
    seg_simp_ary[(seg_ary >= 18) & (seg_ary <= 41)] = 7  # seg vertebrae
    seg_simp_ary[(seg_ary >= 58) & (seg_ary <= 81)] = 8  # seg rib

    return seg_simp_ary


def dcm_to_nifti_manual(source: Path, target: Path) -> Tuple[sitk.Image, np.ndarray]:
    # Load all the DICOM files from a single folder into a list of 3D images and return the numpy array by simpleITK
    # reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(patientDicomPath)
    # reader.SetFileNames(dicom_names)
    # img_itk = reader.Execute()
    # img_3dnp = sitk.GetArrayFromImage(img_itk)

    dicom_files = list(source.rglob('*.dcm'))
    slices = [pydicom.dcmread(file) for file in dicom_files]
    slices = [s for s in slices if s.Modality == 'CT']
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        img_3d = np.stack([s.pixel_array for s in slices])
    except ValueError as e:
        if 'all input arrays must have the same shape' in str(e):
            info(f"pixel must have the same shape, but not on folder {source}\nForce clip by {IMG_SIZE}")
            info([s.pixel_array.shape for s in slices])
            img_3d = np.stack([np.resize(s.pixel_array, (IMG_SIZE, IMG_SIZE)) for s in slices])
        else:
            raise e
    img_3d = img_3d * slices[0].RescaleSlope + slices[0].RescaleIntercept

    affine3d = {
        'spacing': (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[0].SliceThickness),
        'origin': slices[0].ImagePositionPatient,
        'direction': (1, 0, 0, 0, 1, 0, 0, 0, -1)
    }

    img_itk = toITK(img_3d, target, affine3d)
    return img_itk, img_3d


def deal_input(source: Path, target: Path, infoLv: int = 0) -> Tuple[np.ndarray, sitk.Image, Path]:
    if '.nii' not in target.name:
        img_nii = target / (source.name + '.nii.gz')
    else:
        img_nii = target
    if source.is_dir():
        if not img_nii.exists():
            try:
                dcm_to_nifti(source, img_nii)
                info(f'convert {source.name} folder to {img_nii.name}', show=infoLv > 0)
                img_itk = sitk.ReadImage(str(img_nii))
            except Exception as e:
                info('defalut convert method failed, try manual convert method', show=infoLv > 0)
                info(e, show=infoLv > 1)
                img_itk, img_3d = dcm_to_nifti_manual(source, img_nii)
                info(f'convert {source.name} folder to {img_nii.name}', show=infoLv > 0)
        else:
            img_itk = sitk.ReadImage(str(img_nii))
    elif source.is_file() and '.nii' in source.name:
        img_itk = sitk.ReadImage(str(source))
        img_nii = source
    else:
        raise ValueError(f'input {source} not a valid file or folder')
    img_3d = sitk.GetArrayFromImage(img_itk)
    return img_3d, img_itk, img_nii


def pre_seg_pred(img_path: Path, target: Path, gpu: bool = False, infoLv: int = 0) -> Tuple[np.ndarray, sitk.Image, Path]:
    if '.nii' not in target.name:
        seg_nii = target / (img_path.name + '_seg.nii.gz')
    elif 'seg' not in target.name:
        seg_nii = target.parent / (target.name.replace('.nii', '_seg.nii'))
    else:
        seg_nii = target
    with HiddenPrints():
        totalsegmentator(img_path, seg_nii, ml=True, fast=not gpu)
    seg_itk = sitk.ReadImage(str(seg_nii))
    seg_simp_ary = seg_simplify(seg_itk)
    seg_simp_itk = toITK(seg_simp_ary, seg_nii, {
        'spacing': seg_itk.GetSpacing(),
        'origin': seg_itk.GetOrigin(),
        'direction': seg_itk.GetDirection()
    })
    info('pre-segmentation is done', show=infoLv > 0)

    return seg_simp_ary, seg_simp_itk, seg_nii


def preprocess(img3d: np.ndarray,
               seg3d: np.ndarray,
               window: Tuple[int, int],
               xfmComp: Callable,
               infoLv: int = 0) -> Tuple[torch.Tensor, Tuple[int, int]]:
    lung_slice = [i for i, ls in enumerate(seg3d) if np.sum(ls == 5) > 0]
    lung_filter = (min(lung_slice), max(lung_slice))
    info(f'lung filter: {lung_filter}', show=infoLv > 1)
    img_md = img3d.copy()[lung_filter[0]:lung_filter[1] + 1]
    img_md = np.clip(img_md, window[0], window[1])  # windowing
    img_md = img_md.astype(np.float32)
    img_md = (img_md - window[0]) / (window[1] - window[0])  # normalize to [0, 1]

    img4d_ts = torch.stack([xfmComp(img_md[i]) for i in range(img_md.shape[0])])
    info(f'preprocess shape: {img4d_ts.shape}', show=infoLv > 0)
    return img4d_ts, lung_filter


def load_model(net: torch.nn.Module, weightPath: Path, infoLv: int = 0) -> torch.nn.Module:
    if weightPath is not None:
        state_dict = torch.load(weightPath, map_location=torch.device('cpu'))
        moderfied_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):  # 删除"module."前缀
                new_key = key[7:]
            else:
                new_key = key
            moderfied_state_dict[new_key] = value
        del state_dict
        net.load_state_dict(moderfied_state_dict)
        info(f'load model from {weightPath}', show=infoLv > 0)
    return net


def evaluate(net: torch.nn.Module,
             img4d_ts: torch.Tensor,
             batchSize: int = 8,
             gpu: bool = True,
             infoLv: int = 0) -> torch.Tensor:
    torch.cuda.set_device(GPU_NUMBER)
    device = torch.device("cuda" if gpu else "cpu")
    info("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()), show=infoLv > 0)

    net.to(device).eval()
    with torch.no_grad():
        output = torch.cat([net(img4d_ts[i:i + batchSize].to(device)) for i in range(0, img4d_ts.shape[0], batchSize)], dim=0)
    return output


def postprocess(output: torch.Tensor, seg3d: np.ndarray, lung_filter: Tuple[int, int], threshold: float = 0.5, infoLv: int = 0):
    output = output.cpu()
    output = torch.sigmoid(output)
    output = output.numpy()

    ref_seg = seg3d[lung_filter[0]:lung_filter[1] + 1]

    rili_full_flat = np.zeros(ref_seg.shape, dtype=np.uint8)
    diff_size = (IMG_SIZE - NET_SIZE) // 2
    for i, pp in enumerate(output):
        pc, _ = cv2.findContours(np.array(pp[0] > threshold, np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in pc:
            if cv2.contourArea(c) > SMOOTH_AREA:
                temp = np.zeros_like(rili_full_flat[i], dtype=np.uint8)
                cv2.fillPoly(temp, [c + diff_size], 1)

                if np.sum((ref_seg[i] == 5) & temp) / np.sum(temp) > 0.9:
                    cv2.fillPoly(rili_full_flat[i], [c + diff_size], 1)

    return rili_full_flat


def merge_pred(rili: np.ndarray, seg: np.ndarray, lung_filter: Tuple[int, int], infoLv: int = 0) -> np.ndarray:
    rili_whole = np.zeros(seg.shape, dtype=np.uint8)
    rili_whole[lung_filter[0]:lung_filter[1] + 1] = rili

    seg[rili_whole == 1] = 9
    return seg


def draw(img3D: np.ndarray, msk3D: np.ndarray, savePath: Path, window=(-1000, 200)):
    COLORS = ['#FF5733', '#4CAF50', '#4287f5', '#FFC300', '#E040FB', '#FF9933', '#5C5C5C', '#FF66CC', '#00BFFF', '#FF1493']
    pos_pairs = []
    for i, (x, y) in enumerate(zip(img3D, msk3D, strict=True)):
        if np.sum(y == 9) > 0:
            pos_pairs.append((i, x, y))

    d = len(pos_pairs)
    di, dj = [[j, j + i] for j in range(d) for i in range(2) if j**2 + i * j > d][0]
    fig = plt.figure(figsize=(min(dj * 4, 24), min(di * 4, 24)))
    for i, (idx, img, msk) in enumerate(pos_pairs):
        ax = fig.add_subplot(di, dj, i + 1)
        ax.imshow(img, cmap='bone', vmin=window[0], vmax=window[1])
        # ax.imshow(msk, alpha=.3, cmap='rainbow')
        for v in range(1, 10):
            pc, _ = cv2.findContours(np.array(msk == v, np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for c in pc:
                plt.plot(c[:, 0, 0], c[:, 0, 1], color=COLORS[v])
        ax.set_title(f"slice {idx}, area {np.sum(msk)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(savePath))
    plt.show()
    plt.close()
    return


def inference2itk(source: str | Path,
                  targetFolder: str | Path,
                  modelWeight: str | Path,
                  type: str = 'nrrd',
                  batchSize: int = BATCH_SIZE,
                  window: Tuple[int, int] = HU_CLIP,
                  xfmComp: Callable = XFM_COMP,
                  net: torch.nn.Module = NET,
                  preview: bool = False,
                  infoLv: int = 1):
    start = time.time()

    if isinstance(source, str):
        source = Path(source)
    if isinstance(targetFolder, str):
        targetFolder = Path(targetFolder)
        targetFolder.mkdir(parents=True, exist_ok=True)
    if isinstance(modelWeight, str):
        modelWeight = Path(modelWeight)

    img3d, img_itk, img_nii = deal_input(source=source, target=targetFolder, infoLv=infoLv)
    seg3d, seg_itk, seg_nii = pre_seg_pred(img_nii, target=targetFolder, gpu=torch.cuda.is_available(), infoLv=infoLv)
    img4dts, lfilter = preprocess(img3d, seg3d, window=HU_CLIP, xfmComp=xfmComp, infoLv=infoLv)
    net = load_model(net, modelWeight, infoLv=infoLv)
    rili_raw = evaluate(net, img4dts, batchSize=batchSize, gpu=torch.cuda.is_available(), infoLv=infoLv)
    rili = postprocess(rili_raw, seg3d, lfilter, threshold=THRESHOLD, infoLv=infoLv)
    seg_rili = merge_pred(rili, seg3d, lfilter, infoLv=infoLv)

    rili_nii = targetFolder / (source.name + '_rili.nii.gz')
    seg_rili_itk = toITK(seg_rili, rili_nii, {
        'spacing': seg_itk.GetSpacing(),
        'origin': seg_itk.GetOrigin(),
        'direction': seg_itk.GetDirection()
    })

    if type == 'nrrd':
        os.remove(img_nii)
        os.remove(seg_nii)
        os.remove(rili_nii)
        sitk.WriteImage(img_itk, targetFolder / (source.name + '.nrrd'))
        sitk.WriteImage(seg_itk, targetFolder / (source.name + 'seg.nrrd'))
        sitk.WriteImage(seg_rili_itk, targetFolder / (source.name + 'rili.seg.nrrd'))

    if preview:
        draw(img3d, seg_rili, targetFolder / (source.name + '_seg.png'))
        info(f'Preview saved: {source.name} -> {targetFolder / (source.name + "_seg.png")}')

    info(f'Inference finished: {source.name} -> {targetFolder}, time cost: {time.time() - start:.2f}s')


class ArgsRiliTotalSeg(argparse.Namespace):
    net: str = "unet"
    '''Network model (default: unet)'''
    weightPath: str = "0.0244_109.pth"
    '''Network parameters'''
    input: str
    '''input data dcms folder or nii file'''
    output: str
    '''output folder'''
    batchSize: int = 8
    '''Batch size for computation'''
    window: tuple = (-1000, 200)
    '''Lung windows range for preview images'''
    preview: bool = False
    '''preview images'''
    infoLv: int = 1
    '''info level 0: no info, 1: only important info, 2: all info'''
    type: str = 'nrrd'
    '''output type: nrrd or nii'''


if __name__ == "__main__":
    default_weight_path = (Path(__file__).parent / "D.pth").as_posix()

    parser = argparse.ArgumentParser(description="RILI total segmentator")
    parser.add_argument('-n', "--net", type=str, default="unet+", help="Network model (default: unet+)")
    parser.add_argument('-w', "--weightPath", type=str, default=default_weight_path, help="Network parameters")
    parser.add_argument('-i', "--input", type=str, required=True, help="input data dcms folder or nii file")
    parser.add_argument('-o', "--output", type=str, required=True, help="output folder")
    parser.add_argument('-b', "--batchSize", type=int, default=8, help="Batch size for computation")
    parser.add_argument('-r', "--window", type=int, nargs=2, default=(-1000, 200), help="Lung windows range for preview images")
    parser.add_argument('-p', "--preview", action="store_true", help="preview images")
    parser.add_argument('-l',
                        "--infoLv",
                        type=int,
                        default=1,
                        help="info level 0: no info, 1: only important info, 2: all info")
    parser.add_argument('-t', "--type", type=str, default='nrrd', help="output type: nrrd or nii")
    args = ArgsRiliTotalSeg()
    parser.parse_args(namespace=args)

    inference2itk(args.input,
                  args.output,
                  args.weightPath,
                  batchSize=args.batchSize,
                  window=args.window,
                  preview=args.preview,
                  infoLv=args.infoLv)
