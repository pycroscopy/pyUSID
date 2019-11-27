# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
from PIL import Image
import h5py
import numpy as np
from .data_utils import validate_aux_dset_pair, delete_existing_file
sys.path.append("../../pyUSID/")
from pyUSID.io import ImageTranslator, hdf_utils, USIDataset
from pyUSID.io.image import read_image

if sys.version_info.major == 3:
    unicode = str
else:
    FileExistsError = ValueError
    FileNotFoundError = ValueError

image_path = 'random_image.png'
rand_image = np.uint16(np.random.randint(0, high=255, size=(128, 256)))


class TestImage(unittest.TestCase):

    def setUp(self):
        result = Image.fromarray(rand_image.astype(np.uint8))
        for file_path in [image_path, image_path.replace('.png', '.h5')]:
            delete_existing_file(file_path)
        result.save(image_path)

    def tearDown(self):
        delete_existing_file(image_path)


class TestReadImage(TestImage):

    def test_color_to_bw_image(self):
        color_image_path = './tests/io/logo_v01.png'
        img_obj = Image.open(color_image_path).convert(mode="L")
        pillow_obj = read_image(color_image_path, as_numpy_array=False)
        self.assertEqual(img_obj, pillow_obj)

    def test_color(self):
        color_image_path = './tests/io/logo_v01.png'
        img_obj = Image.open(color_image_path)
        pillow_obj = read_image(color_image_path, as_numpy_array=False, as_grayscale=False)
        self.assertEqual(img_obj, pillow_obj)

    def test_text_to_numpy_simple(self):
        img_data = rand_image.astype(np.uint8)
        img_path = 'image_text.txt'
        delete_existing_file(img_path)
        np.savetxt(img_path, img_data)
        np_data = read_image(image_path, as_numpy_array=True)
        self.assertIsInstance(np_data, np.ndarray)
        self.assertTrue(np.allclose(np_data, img_data))
        delete_existing_file(img_path)

    def test_text_to_numpy_complex(self):
        img_data = np.uint16(np.random.randint(0, high=255, size=(4, 3)))
        img_path = 'image_text.csv'
        delete_existing_file(img_path)
        txt_kwargs = {'delimiter': ',',
                      'newline': '\n',
                      'header':  'cat, dog, cow'}
        np.savetxt(img_path, img_data, **txt_kwargs)
        np_data = read_image(img_path, as_numpy_array=True, delimiter=',', skiprows=1)
        self.assertIsInstance(np_data, np.ndarray)
        self.assertTrue(np.allclose(np_data, img_data))
        delete_existing_file(img_path)

    def test_text_complex_to_pillow(self):
        img_data = np.uint16(np.random.randint(0, high=255, size=(4, 3)))
        img_path = 'image_text.csv'
        delete_existing_file(img_path)
        txt_kwargs = {'delimiter': ',',
                      'newline': '\n',
                      'header':  'cat, dog, cow'}
        np.savetxt(img_path, img_data, **txt_kwargs)
        pillow_obj = read_image(img_path, as_grayscale=True, as_numpy_array=False,
                                delimiter=',', skiprows=1)
        self.assertIsInstance(pillow_obj, Image.Image)
        self.assertTrue(np.allclose(np.asarray(pillow_obj), img_data))
        delete_existing_file(img_path)

    def test_to_numpy(self):
        np_data = read_image(image_path, as_numpy_array=True)
        self.assertIsInstance(np_data, np.ndarray)
        self.assertTrue(np.allclose(np_data, rand_image))

    def test_to_pillow(self):
        pillow_obj = read_image(image_path, as_numpy_array=False)
        self.assertIsInstance(pillow_obj, Image.Image)
        self.assertTrue(np.allclose(np.asarray(pillow_obj), rand_image))


class TestImageTranslator(TestImage):

    def basic_file_validation(self, h5_f):
        self.assertEqual('ImageTranslator', hdf_utils.get_attr(h5_f, 'translator'))

        # First level should have absolutely nothing besides one group
        self.assertEqual(len(h5_f.items()), 1)
        self.assertTrue('Measurement_000' in h5_f.keys())
        h5_meas_grp = h5_f['Measurement_000']
        self.assertIsInstance(h5_meas_grp, h5py.Group)

        # Again, this group should only have one group - Channel_000
        self.assertEqual(len(h5_meas_grp.items()), 1)
        self.assertTrue('Channel_000' in h5_meas_grp.keys())
        h5_chan_grp = h5_meas_grp['Channel_000']
        self.assertIsInstance(h5_chan_grp, h5py.Group)

        # This channel group is not expected to have any (custom) attributes but it will contain the main dataset
        self.assertEqual(len(h5_chan_grp.items()), 5)
        for dset_name in ['Raw_Data', 'Position_Indices', 'Position_Values', 'Spectroscopic_Indices',
                          'Spectroscopic_Values']:
            self.assertTrue(dset_name in h5_chan_grp.keys())
            h5_dset = h5_chan_grp[dset_name]
            self.assertIsInstance(h5_dset, h5py.Dataset)

        usid_main = USIDataset(h5_chan_grp['Raw_Data'])

        self.assertIsInstance(usid_main, USIDataset)
        self.assertEqual(usid_main.name.split('/')[-1], 'Raw_Data')
        self.assertEqual(usid_main.parent, h5_chan_grp)

        validate_aux_dset_pair(self, h5_chan_grp, usid_main.h5_spec_inds, usid_main.h5_spec_vals, ['arb'],
                               ['a.u.'], np.atleast_2d([0]), h5_main=usid_main, is_spectral=True)

    def main_translate(self, **kwargs):

        h5_path = kwargs.pop('h5_path', image_path.replace('.png', '.h5'))
        delete_existing_file(h5_path)

        input_image = rand_image.copy()
        usize, vsize = input_image.shape[:2]

        translator = ImageTranslator()
        h5_path = translator.translate(image_path, **kwargs)

        image_parms = dict()

        if 'bin_factor' in kwargs.keys():
            bin_factor = kwargs.pop('bin_factor')
            if bin_factor is None:
                _ = kwargs.pop('interp_func', None)
            else:
                if isinstance(bin_factor, int):
                    bin_factor = (bin_factor, bin_factor)
                interp_func = kwargs.pop('interp_func', Image.BICUBIC)

                image_parms.update({'image_binning_size': np.array(bin_factor),
                                    'image_PIL_resample_mode': interp_func})

                img_obj = Image.fromarray(input_image)
                img_obj = img_obj.convert(mode="L")
                img_obj = img_obj.resize((int(vsize / bin_factor[1]), int(usize / bin_factor[0])),
                                         resample=interp_func)
                input_image = np.asarray(img_obj)

        image_parms.update({'normalized': False})
        input_image = input_image.copy()
        if 'normalize' in kwargs.keys():
            normalize = kwargs.pop('normalize')
            if normalize:
                input_image -= np.min(input_image)
                input_image = input_image / np.float32(np.max(input_image))
                image_parms.update({'normalized': True})

        image_parms.update({'image_min': np.min(input_image), 'image_max': np.max(input_image)})
        with h5py.File(h5_path, mode='r') as h5_f:

            self.basic_file_validation(h5_f)

            h5_meas_grp = h5_f['Measurement_000']
            h5_chan_grp = h5_meas_grp['Channel_000']
            usid_main = USIDataset(h5_chan_grp['Raw_Data'])

            # check the attributes under this group
            for key, expected_val in image_parms.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_meas_grp, key) == expected_val))

            one_d_image = input_image.T.reshape(-1, 1)
            self.assertTrue(np.allclose(one_d_image, usid_main[()]))
            # self.assertTrue(np.allclose(rand_image, np.reshape(usid_main[()], rand_image.shape)))

            pos_data = np.vstack((np.tile(np.arange(input_image.shape[0]), input_image.shape[1]),
                                  np.repeat(np.arange(input_image.shape[1]), input_image.shape[0]))).T

            validate_aux_dset_pair(self, h5_chan_grp, usid_main.h5_pos_inds, usid_main.h5_pos_vals, ['Y', 'X'],
                                   ['a.u.', 'a.u.'], pos_data, h5_main=usid_main, is_spectral=False)

        delete_existing_file(h5_path)

    def test_basic_translate(self):
        self.main_translate()


class TestBinning(TestImageTranslator):

    def test_single_default_interp(self):
        self.main_translate(bin_factor=2)

    def test_tuple_default_interp(self):
        self.main_translate(bin_factor=(1, 2))

    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, bin_factor=(1, 2, 3))

    def test_neg_parms(self):
        with self.assertRaises(ValueError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, bin_factor=-2)

    def test_float_parms(self):
        with self.assertRaises(TypeError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, bin_factor=1.34)

    def test_invalid_dtype(self):
        with self.assertRaises(TypeError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, bin_factor=['dfrdd', True])

    def test_custom_interp(self):
        self.main_translate(bin_factor=2, interp_func=Image.NEAREST)

    def test_invalid_interp(self):
        with self.assertRaises(ValueError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, bin_factor=2, interp_func='dsdsdsd')


class TestNormalization(TestImageTranslator):

    def test_normalize_only(self):
        self.main_translate(normalize=True)

    def test_normalize_and_default_interp(self):
        self.main_translate(normalize=True, bin_factor=2)


class TestFile(TestImageTranslator):

    def test_invalid_h5_path(self):
        with self.assertRaises(TypeError):
            translator = ImageTranslator()
            _ = translator.translate(image_path, h5_path=np.arange(4))

    def test_path_not_str(self):
        with self.assertRaises(TypeError):
            translator = ImageTranslator()
            _ = translator.translate(np.arange(4))

    def test_path_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            translator = ImageTranslator()
            _ = translator.translate('no_such_file.png')

    def test_output_h5_file_already_exists(self):
        with h5py.File(image_path.replace('.png', '.h5'), mode='w') as _:
            pass
        with self.assertRaises(FileExistsError):
            translator = ImageTranslator()
            _ = translator.translate(image_path)

    def test_valid_h5_path(self):
        self.main_translate(h5_path='custom_path.h5')
        self.main_translate(h5_path='custom_path.txt')


if __name__ == '__main__':
    unittest.main()
