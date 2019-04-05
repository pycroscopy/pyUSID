# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
from PIL import Image
import h5py
import numpy as np
from .data_utils import validate_aux_dset_pair
sys.path.append("../../pyUSID/")
from pyUSID.io import ImageTranslator, write_utils, hdf_utils, USIDataset

if sys.version_info.major == 3:
    unicode = str

image_path = 'random_image.png'
rand_image = np.random.randint(0, high=255, size=(128, 256))


class TestImageTranslator(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def setUp(self):
        result = Image.fromarray(rand_image.astype(np.uint8))
        result.save(image_path)

    def tearDown(self):
        self.__delete_existing_file(image_path)

    def test_translate(self):

        h5_path = image_path.replace('.png', '.h5')
        self.__delete_existing_file(h5_path)

        translator = ImageTranslator()
        h5_path = translator.translate(image_path, h5_path=None,
                                       bin_factor=None, interp_func=Image.BICUBIC,
                                       normalize=False)

        with h5py.File(h5_path, mode='r') as h5_f:
            # we are not interested in most of the attributes under root besides two:
            self.assertEqual('ImageTranslator', hdf_utils.get_attr(h5_f, 'translator'))

            # First level should have absolutely nothing besides one group
            self.assertEqual(len(h5_f.items()), 1)
            self.assertTrue('Measurement_000' in h5_f.keys())
            h5_meas_grp = h5_f['Measurement_000']
            self.assertIsInstance(h5_meas_grp, h5py.Group)

            # check the attributes under this group
            """
            if use_parm_dict:
                for key, expected_val in attrs.items():
                    self.assertTrue(np.all(hdf_utils.get_attr(h5_meas_grp, key) == expected_val))
            """

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
            one_d_image = rand_image.T.reshape(-1, 1)
            self.assertTrue(np.allclose(one_d_image, usid_main[()]))
            # self.assertTrue(np.allclose(rand_image, np.reshape(usid_main[()], rand_image.shape)))

            pos_data = np.vstack((np.tile(np.arange(rand_image.shape[0]), rand_image.shape[1]),
                                  np.repeat(np.arange(rand_image.shape[1]), rand_image.shape[0]))).T

            validate_aux_dset_pair(self, h5_chan_grp, usid_main.h5_pos_inds, usid_main.h5_pos_vals, ['Y', 'X'],
                                   ['a.u.', 'a.u.'], pos_data, h5_main=usid_main, is_spectral=False)

            validate_aux_dset_pair(self, h5_chan_grp, usid_main.h5_spec_inds, usid_main.h5_spec_vals, ['arb'],
                                   ['a.u.'], np.atleast_2d([0]), h5_main=usid_main, is_spectral=True)

        self.__delete_existing_file(h5_path)


if __name__ == '__main__':
    unittest.main()
