"""Utility functions and dataset classes for event-based data processing and training."""
import os
import numpy as np
import torch
import PIL.Image as Image
from tqdm import tqdm
from torch.utils.data import Dataset
import scipy.special
import cv2
import numba as nb
from skimage import io
from sklearn.linear_model import LinearRegression

from h5 import H5EventsReader
from var import sparse_diff
event_height, event_width = 720, 1280


class EventCountNormalization(object):
    def __init__(self, integration_time_s=1):
        self.integration_time_s = integration_time_s

    def __call__(self, mat):
        num_channels = mat.shape[-1] - 1
        scalar = 1 / 20 * (num_channels / 2) / self.integration_time_s
        mat[..., :-1] *= scalar
        return mat


class EventNoiseCountWrapper(object):
    def __init__(self, num_photon_scalar=50, num_time=100, eps_pos=0.1, eps_neg=0.1, bias_pr=0.0, illum_offset=0.0,
                 constant_noise_neg=0.0, polarity=True, pixel_bin=1, varying_eps=False, poisson_sample=False):
        self.event_obj = EventNoiseCount(num_photon_scalar=num_photon_scalar, num_time=num_time, eps_pos=eps_pos,
                                         eps_neg=eps_neg, bias_pr=bias_pr, illum_offset=illum_offset,
                                         constant_noise_neg=constant_noise_neg, polarity=polarity,
                                         pixel_bin=pixel_bin, output_numpy=True, varying_eps=varying_eps,
                                         poisson_sample=poisson_sample)
        self.pixel_bin = pixel_bin

    def __call__(self, input_mat):
        sample = Image.fromarray(input_mat[:, :, -1]*255)
        out = self.event_obj(sample)
        if self.pixel_bin > 1:
            img = input_mat[:, :, -1].reshape((input_mat.shape[0] // self.pixel_bin, self.pixel_bin, input_mat.shape[1] // self.pixel_bin, self.pixel_bin)).mean(axis=(1, 3))[..., np.newaxis]
        else:
            img = input_mat[:, :, -1, np.newaxis]
        out = np.concatenate([out.transpose((1, 2, 0)), img], axis=-1)
        return out


class EventNoiseCount(object):

    def __init__(self, num_photon_scalar=50, num_time=100, eps_pos=0.1, eps_neg=0.1, bias_pr=0.0, illum_offset=0.0,
                 constant_noise_neg=0.0, polarity=True, pixel_bin=1, output_numpy=False, varying_eps=False,
                 poisson_sample=False):
        self.num_photon_scalar = num_photon_scalar
        self.num_time = num_time
        self.eps_pos = eps_pos
        self.eps_neg = eps_neg
        self.bias_pr = bias_pr
        self.illum_offset = illum_offset
        self.constant_noise_neg = constant_noise_neg
        self.polarity = polarity
        self.pixel_bin = pixel_bin
        self.output_numpy = output_numpy
        self.varying_eps = varying_eps
        self.poission_sample = poisson_sample  # by default sample from negative binomial

        with np.load('lux_measurement.npz') as data:
            list_lux_close = data['list_lux_close']
            list_lux_far = data['list_lux_far']
            list_intensity = data['list_intensity']
        self.reg = LinearRegression().fit(list_intensity.reshape(-1, 1), list_lux_close.reshape(-1, 1))
        self.reg_far = LinearRegression().fit(list_intensity.reshape(-1, 1), list_lux_far.reshape(-1, 1))
        self.illuminance_level = self.reg.predict((np.arange(256) ** 2.2).reshape(-1, 1))
        self.illuminance_level_far = self.reg_far.predict((np.arange(256) ** 2.2).reshape(-1, 1))
        self.illuminance_level = self.illuminance_level / np.max(self.illuminance_level) * np.max(self.illuminance_level_far)

        with np.load('synthetic_param.npz') as f:
            self.negative_binomial_r_pos = f['r_pos']
            self.negative_binomial_r_neg = f['r_neg']
        self.rng = np.random.default_rng(seed=int(torch.randint(2**32, (1,))))

    def __call__(self, sample):

        # convert to grayscale, load into numpy
        im_arr = np.array(sample.convert('L')).astype(np.float32)
        im_arr = im_arr.reshape((sample.size[1], sample.size[0]))

        r_pos = np.interp(im_arr, np.linspace(0, 255, len(self.negative_binomial_r_pos)), self.negative_binomial_r_pos)
        r_neg = np.interp(im_arr, np.linspace(0, 255, len(self.negative_binomial_r_neg)), self.negative_binomial_r_neg)
        im_arr_gamma = np.squeeze(self.illuminance_level[im_arr.astype(np.int32)], axis=-1)

        # actual forward noisy model
        random_scale = self.rng.uniform(0.8, 1.2)  # 0.8, 1.2
        eps_pos = self.eps_pos if not self.varying_eps else self.eps_pos * random_scale
        eps_neg = self.eps_neg if not self.varying_eps else self.eps_neg * random_scale
        # p = (1 - scipy.special.erf(self.eps * ((im_arr_gamma + self.luminance_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(2 * (im_arr_gamma + self.luminance_offset) * self.num_photon_scalar + 1e-9))) / 2
        p_pos = (1 - scipy.special.erf((np.exp(eps_pos) - 1) * ((im_arr_gamma + self.illum_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(
            2 * (im_arr_gamma + self.illum_offset) * self.num_photon_scalar * (1 + np.exp(eps_pos)))))
        p_neg = (1 - scipy.special.erf((np.exp(eps_neg) - 1) * ((im_arr_gamma + self.illum_offset) * self.num_photon_scalar + self.bias_pr) / np.sqrt(
            2 * (im_arr_gamma + self.illum_offset) * self.num_photon_scalar * (1 + np.exp(eps_neg)))))

        if self.poission_sample:
            count_pos = self.rng.poisson((p_pos * self.num_time))
            count_neg = self.rng.poisson((p_neg * self.num_time))
        else:
            count_pos = self.rng.negative_binomial(r_pos, r_pos / (p_pos * self.num_time + r_neg))
            count_neg = self.rng.negative_binomial(r_neg, r_neg / (p_neg * self.num_time + r_neg))

        if self.constant_noise_neg > 0:
            noise_neg = self.constant_noise_neg
            count_neg += self.rng.poisson(noise_neg, size=p_neg.shape)

        if self.polarity:
            total_count = np.stack([count_pos, count_neg], axis=0)
        else:
            total_count = count_pos + count_neg
            total_count = total_count[np.newaxis]

        if self.pixel_bin > 1:
            total_count = total_count.reshape((total_count.shape[0], total_count.shape[1] // self.pixel_bin, self.pixel_bin,
                                               total_count.shape[2] // self.pixel_bin, self.pixel_bin)).mean(axis=(2, 4))
            im_arr = im_arr.reshape((im_arr.shape[0] // self.pixel_bin, self.pixel_bin,
                                     im_arr.shape[1] // self.pixel_bin, self.pixel_bin)).mean(axis=(1, 3))

        if self.output_numpy:
            return total_count.astype(np.float32)
        else:
            return torch.from_numpy(total_count.astype(np.float32)), torch.from_numpy(im_arr.astype(np.float32)[np.newaxis]/255), torch.tensor(1.0, dtype=torch.float32)


class AugmentImageContrast(object):
    def __init__(self, max_scale, min_scale, seed=19358):
        assert (min_scale >= 0) and (max_scale >= min_scale)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.rng = np.random.default_rng(seed)

    def __call__(self, input_mat):
        scale = self.rng.uniform(self.min_scale, self.max_scale)
        base_level = 0  # self.rng.uniform(0, 1 - scale)
        input_mat[:, :, -1] = input_mat[:, :, -1] * scale + base_level
        return input_mat


@nb.jit(nopython=True)
def count_events(events_, dim_xy_):
    count = np.zeros((dim_xy_[1], dim_xy_[0]), dtype=np.float32)
    for e in events_:
        count[e[1], e[0]] += 1
    return count


def event_interval_std(events, dim_xy_, integration_time_s):
    event_std = np.ones((dim_xy_[1], dim_xy_[0]), dtype=np.float32) * integration_time_s

    events_reorg = [[[] for _ in range(dim_xy_[1])] for _ in range(dim_xy_[0])]
    for e in events:
        events_reorg[e[0]][e[1]].append(e[3])

    min_time = events[0][3]
    for x in range(dim_xy_[0]):
        for y in range(dim_xy_[1]):
            t = np.array(events_reorg[x][y])
            if len(t) > 2:
                t[1:] = t[1:] - t[:-1]
                event_std[y, x] = np.std(t[1:] - t[:-1]) * 1e-6
            elif len(t) == 2:
                t -= min_time
                t[1] -= t[0]
                event_std[y, x] = np.std(t) * 1e-6
            elif len(t) == 1:
                event_std[y, x] = 0.5 * integration_time_s

    return event_std


class RawEventCountData(Dataset):
    def __init__(self, data_folder, dim_xy=(1280, 720), n_limit=None):
        super().__init__()
        _image_files = [f for f in sorted(os.listdir(data_folder)) if f.endswith('.npy')]

        event_counts, image_files = [], []

        for i in tqdm(range(len(_image_files) if n_limit is None else n_limit)):
            events = np.load(os.path.join(data_folder, _image_files[i]))
            event_counts.append(count_events(events, dim_xy))
            image_files.append(_image_files[i])

        self.event_counts = torch.from_numpy(np.stack(event_counts))
        self.files = image_files
        self.folder = data_folder

        print(f'Loaded {len(self)} events from {self.folder}')
        print(self.event_counts.shape)

    def preload(self, device='cuda:0'):
        self.event_counts = self.event_counts.to(device)
        return self

    def astype(self, dtype):
        self.event_counts = self.event_counts.type(dtype)
        return self

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.event_counts[idx]


class EventImagePairDataset(Dataset):
    def __init__(self, image_folder, event_folder,
                 integration_time_s=1, total_time_s=10, start_time_s=-1, time_bin=1, pixel_bin=1,
                 polarity=False, std_channel=False, n_limit=None, transform=None,
                 img_suffix='.jpg', calib_img_path=None, image_names=None):
        """
        :param image_folder: folder containing images
        :param event_folder: folder containing event recordings
        :param integration_time_s: integration time in seconds, if -1, randomly sample from [1, total_time_s]
        :param total_time_s: total time of the event recording in seconds
        :param start_time_s: start time of the event recording in seconds, if -1, randomly sample from [0, total_time_s - integration_time_s]
        :param time_bin: number of time bins to split the integration time into (for temporal resolution)
        :param pixel_bin: average pooling kernel size, if 1, no pooling
        :param polarity: whether to count polarity events separately
        :param std_channel: whether to add a channel of event interval standard deviation
        :param n_limit: limit the number of images to load
        :param transform: transform to apply to the image and event count matrix for data augmentation
        :param img_suffix: image file suffix. Default is '.jpg'
        :param calib_img_path: path to the calibration image
        :param image_names: optional filenames to load instead of scanning the image folder
        """
        super().__init__()
        _image_files = ([f for f in sorted(os.listdir(image_folder)) if f.endswith(img_suffix)]
                        if image_names is None else list(image_names))
        image_files, images, event_files, event_images = [], [], [], []
        self.event_dim_xy = [1280, 720]
        self.output_dim_xy = [1280 // pixel_bin, 720 // pixel_bin]
        self.integration_time_s = integration_time_s
        self.total_time_s = total_time_s
        self.start_time_s = start_time_s
        self.time_bin = time_bin
        self.pixel_bin = pixel_bin
        self.polarity = polarity
        self.std_channel = std_channel
        self.transform = transform

        for i in tqdm(range(len(_image_files) if n_limit is None else n_limit)):
            image = io.imread(os.path.join(image_folder, _image_files[i]))

            if image.shape[:2] == (1080, 1920):
                image_files.append(_image_files[i])
                event_files.append(os.path.join(event_folder, _image_files[i].split('/')[-1].replace(img_suffix, '.h5')))
            else:
                print(f'Image {_image_files[i]} has invalid shape {image.shape}')
        self.image_files = image_files
        self.event_files = event_files
        self.folder = image_folder
        self.event_folder = event_folder

        if calib_img_path is not None:
            calib_img = cv2.resize(cv2.imread(calib_img_path), (1280, 720))
            calib_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
            calib_img = cv2.equalizeHist(calib_img)
            calib_img[calib_img<127]=0
            calib_img[calib_img>=127]=255
            self.H_inv = calibrate_distortion(io.imread(calib_img_path))
        else:
            self.H_inv = H_inv

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        integration_time_s = self.integration_time_s if self.integration_time_s > 0 \
            else np.clip(np.random.normal(1.0, 4), 1.0, self.total_time_s).astype(np.float32)

        events = self.load_events(self.event_files[idx], integration_time_s)
        if not len(events):
            raise ValueError(f'No events in the requested interval for {self.event_files[idx]}')
        time_bin_start = np.linspace(events['t'][0], events['t'][-1], self.time_bin + 1)[:-1]
        bin_ind = np.searchsorted(events['t'], time_bin_start, side='right')
        events['x'] = events['x'] // self.pixel_bin
        events['y'] = events['y'] // self.pixel_bin
        count_bin = np.zeros((self.output_dim_xy[1], self.output_dim_xy[0],
                              self.time_bin * 2 if self.polarity else self.time_bin), dtype=np.float32)
        for i in range(self.time_bin):
            e = events[bin_ind[i]:(bin_ind[i + 1] if i + 1 < self.time_bin else None)]

            if self.polarity:
                count_pos = undistort(count_events(e[e['p'] == 1], nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_neg = undistort(count_events(e[e['p'] == 0], nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_bin[:, :, i * 2] = count_pos / self.pixel_bin / self.pixel_bin
                count_bin[:, :, i * 2 + 1] = count_neg / self.pixel_bin / self.pixel_bin
            else:
                count = undistort(count_events(e, nb.typed.List(self.output_dim_xy)), self.H_inv, self.output_dim_xy)
                count_bin[:, :, i] = count / self.pixel_bin / self.pixel_bin

        image = self.load_image(os.path.join(self.folder, self.image_files[idx]))

        if self.std_channel:
            event_std = sparse_diff(x=torch.from_numpy(events['y'].astype(np.int64)),
                                    y=torch.from_numpy(events['x'].astype(np.int64)),
                                    t=torch.from_numpy(events['t'].astype(np.int64)),
                                    shape=(self.output_dim_xy[1], self.output_dim_xy[0]))
            event_std = undistort(event_std.numpy()**0.5 * 1e-6, self.H_inv, self.output_dim_xy)
            count_bin = np.concatenate([count_bin, event_std[..., np.newaxis]], axis=-1)

        combined = np.concatenate([count_bin, image[..., np.newaxis]], axis=-1)
        if self.transform is not None:
            combined = self.transform(combined)

        return torch.from_numpy(combined[..., :-1].copy().transpose((2, 0, 1))), torch.from_numpy(combined[..., -1:].copy().transpose((2, 0, 1))), torch.tensor(integration_time_s, dtype=torch.float32)

    def load_events(self, event_file_path, integration_time_s):
        h = H5EventsReader(event_file_path)
        start_us = self.start_time_s * 1e6
        if self.start_time_s == -1:
            start_us = torch.randint(0, int((self.total_time_s - integration_time_s) * 1e6 + 1), (1,)).item()

        try:
            events = h.read_interval(start_us, start_us + integration_time_s * 1e6)
        except Exception as e:
            print(f'Error reading {event_file_path}, start {start_us}')
            if integration_time_s < 3:
                start_us = (start_us + 2 * integration_time_s * 1e6) % int((self.total_time_s - integration_time_s) * 1e6)
            else:
                start_us = torch.randint(0, int((self.total_time_s - integration_time_s) * 1e6), (1,)).item()

            print('Trying again with start', start_us)
            events = h.read_interval(start_us, start_us + integration_time_s * 1e6)
        return events

    def load_image(self, image_file_path):
        image = io.imread(image_file_path, as_gray=True)
        image = cv2.resize(image, (self.event_dim_xy[0] // self.pixel_bin, self.event_dim_xy[1] // self.pixel_bin))
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        return image.astype(np.float32)


def data_split(dataset, validation_split=0.1, testing_split=0.2, seed=42):
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    test_size = int(testing_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset, test_dataset


def calibrate_distortion(calib_img):
    gt_img = cv2.resize(cv2.imread('./calibration.png'), (1280, 720))
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    checker_size = (37, 20)
    ret, corners = cv2.findChessboardCorners(gt_img, checker_size)

    ret_m, corners_m = cv2.findChessboardCorners(calib_img, checker_size)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners2 = cv2.cornerSubPix(gt_img, corners, (11, 11), (-1, -1), criteria)
    corners_m2 = cv2.cornerSubPix(calib_img, corners_m, (11, 11), (-1, -1), criteria)

    # Compute the homography matrix
    H, _ = cv2.findHomography(corners2, corners_m2)

    # To reverse the transformation on another image (for example, 'm')
    # First, invert the homography matrix
    H_inv = np.linalg.inv(H)

    return H_inv


H_inv = np.array(
    [[1.05374157e+00, 1.29773432e-02, -4.31484473e+01],
     [-1.14301830e-03, 1.08660669e+00, -2.24931360e+01],
     [-1.49652303e-05, 3.06551912e-05, 9.99973086e-01]])


def undistort(img, H_inv_, dim_xy=(1280, 720)):
    H_inv_cur = H_inv_.copy()
    H_inv_cur[2, :2] = H_inv_cur[2, :2] * 720 / dim_xy[1]
    H_inv_cur[:2, 2] = H_inv_cur[:2, 2] * dim_xy[1] / 720
    return cv2.warpPerspective(img, H_inv_cur, dsize=dim_xy, flags=cv2.INTER_NEAREST).astype(np.float32)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    image = np.array(image)
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def event_image_means(event_arr, grid_shape=(event_height, event_width)):
    out = np.zeros(grid_shape)
    for j in event_arr:
        out[j["y"], j["x"]] += 1
    return out

