"""
Created on Thurs Jun  27 2019

@author: Emily Costa
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pyUSID.viz import plot_utils


"""
class TestGridDecoration(unittest.TestCase):

    #plot_utils.get_plot_grid_size
    def test_get_plot_grid_size(self):
        pass

    def test_get_plot_grid_size_num_plots_error(self):

        #with self.assertRaises(ValueError):
            #num_plots should be < 0
        pass

    def test_get_plot_grid_size_fewer_rows_false(self):
        #tall and narrow grid
        pass

    #plot_utils.set_tick_font_size
    def test_fontsize_not_num(self):
        pass

    def test_fontsize(self):
        pass

    #plot_util.set_tick_font_size.__set_axis_tick
    def test_not_axes(self):
        pass

    def test_complete(self):
        pass

    class TestPlotParams(unittest.TestCase):

        def test_reset_plot_params(self):
            pass

        def test_use_nice_plot_params(self):
            pass    

    def test_is_x_true(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        plot_utils.use_scientific_ticks(axis, is_x=True)

    def test_is_x_false(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        plot_utils.use_scientific_ticks(axis, is_x=False)
"""


class TestUseScientificTicks(unittest.TestCase):

    def test_axis_not_axes(self):
        not_axis = 1
        with self.assertRaises(TypeError):
            plot_utils.use_scientific_ticks(not_axis)

    """
    def test_is_x_not_boolean(self):
        not_bool = 'hello'
        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(TypeError):
            plot_utils.use_scientific_ticks(axis, is_x=not_bool)

    def test_formatting_not_string(self):
        notStr = 55
        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(TypeError):
            plot_utils.use_scientific_ticks(axis, formatting = notStr)
    """


class TestMakeScalarMappable(unittest.TestCase):

    def test_vmin_not_num(self):
        notNum =  'hello'
        with self.assertRaises(AssertionError):
            plot_utils.make_scalar_mappable(notNum, 5)

    def test_vmax_not_num(self):
        notNum =  'hello'
        with self.assertRaises(AssertionError):
            plot_utils.make_scalar_mappable(5, notNum)

    def test_vmin_more_vmax(self):
        with self.assertRaises(AssertionError):
            plot_utils.make_scalar_mappable(5, 3)

    def test_cmap_not_none_wrong_input(self):
        with self.assertRaises(ValueError):
            plot_utils.make_scalar_mappable(3, 5, cmap='hello')

    """
    def test_cmap_none(self):
        plot_utils.make_scalar_mappable(3, 5, cmap=None)

    def test_cmap_not_none(self):
        jet = plt.get_cmap('jet')
        plot_utils.make_scalar_mappable(3, 5, cmap=jet)
    """


class TestCmapFromRGBA(unittest.TestCase):

    def test_name_not_string(self):
        hot_desaturated = [(255.0, (255, 76, 76, 255)),
                           (218.5, (107, 0, 0, 255)),
                           (182.1, (255, 96, 0, 255)),
                           (145.6, (255, 255, 0, 255)),
                           (109.4, (0, 127, 0, 255)),
                           (72.675, (0, 255, 255, 255)),
                           (36.5, (0, 0, 91, 255)),
                           (0, (71, 71, 219, 255))]
        with self.assertRaises(TypeError):
            plot_utils.cmap_from_rgba(5, hot_desaturated, 255)

    def test_interp_vals_not_tuple(self):
        with self.assertRaises(TypeError):
            plot_utils.cmap_from_rgba('cmap', 'hello', 255)

    def test_normalization_val_not_number(self):
        hot_desaturated = [(255.0, (255, 76, 76, 255)),
                           (218.5, (107, 0, 0, 255)),
                           (182.1, (255, 96, 0, 255)),
                           (145.6, (255, 255, 0, 255)),
                           (109.4, (0, 127, 0, 255)),
                           (72.675, (0, 255, 255, 255)),
                           (36.5, (0, 0, 91, 255)),
                           (0, (71, 71, 219, 255))]
        with self.assertRaises(TypeError):
            plot_utils.cmap_from_rgba('cmap', hot_desaturated, 'hi')


class TestMakeLinearAlphaCmap(unittest.TestCase):

    """
    def test_make_linear_alpha_cmap(self):
        solid_color = plt.cm.jet(0.8)
        plot_utils.make_linear_alpha_cmap('my_map', solid_color, 1, min_alpha=0, max_alpha=1)
    """

    def test_name_not_str(self):
        solid_color = plt.cm.jet(0.8)
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap(5, solid_color, 1, min_alpha=0, max_alpha=1)

    def test_solid_color_not_tuple(self):
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap('cmap', 'hello', 1, min_alpha=0, max_alpha=1)

    def test_solid_color_len_wrong(self):
        solid_color = [0, 255, 45]
        with self.assertRaises(ValueError):
            plot_utils.make_linear_alpha_cmap('cmap', solid_color, 1, min_alpha=0, max_alpha=1)

    def test_solid_color_list_not_nums(self):
        solid_color = [0, 255, 'hello', 55]
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap(5, solid_color, 1, min_alpha=0, max_alpha=1)

    def test_solid_normalization_val_not_num(self):
        solid_color = plt.cm.jet(0.8)
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap('cmap', solid_color, 'hello', min_alpha=0, max_alpha=1)

    def test_min_alpha_not_num(self):
        solid_color = plt.cm.jet(0.8)
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap('cmap', solid_color, 1, min_alpha='hello', max_alpha=1)

    def test_max_alpha_not_num(self):
        solid_color = plt.cm.jet(0.8)
        with self.assertRaises(TypeError):
            plot_utils.make_linear_alpha_cmap('cmap', solid_color, 1, min_alpha=0, max_alpha='hello')

    def test_max_less_than_min_alpha(self):
        solid_color = plt.cm.jet(0.8)
        with self.assertRaises(ValueError):
            plot_utils.make_linear_alpha_cmap('cmap', solid_color, 1, min_alpha=1, max_alpha=0)


class TestDiscreteCmap(unittest.TestCase):

    """
    def test_cmap_is_None(self):
        plot_utils.discrete_cmap(num_bins=5)


    def test_cmap_is_not_None(self):
        plot_utils.discrete_cmap(num_bins=5, cmap=plt.get_cmap('jet'))
    """

    def test_numbins_is_not_uint(self):
        with self.assertRaises(TypeError):
            plot_utils.discrete_cmap(num_bins='hello')

    def test_cmap_not_str(self):
        with self.assertRaises(ValueError):
            plot_utils.discrete_cmap(num_bins=1, cmap='hello')


class TestGetCMapObject(unittest.TestCase):

    def test_cmap_not_cmap(self):
        with self.assertRaises(ValueError):
            plot_utils.get_cmap_object(cmap='hello')

    def test_none(self):
        self.assertEqual(plt.cm.viridis, plot_utils.get_cmap_object(None))

    def test_string_name(self):
        self.assertEqual(plt.cm.jet, plot_utils.get_cmap_object(plt.get_cmap('jet')))

    def test_wrong_dtype(self):
        with self.assertRaises(TypeError):
            plot_utils.get_cmap_object(5)


class TestRainbowPlot(unittest.TestCase):

    def test_axis_not_axis(self):
        notAxis = 5
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)
        with self.assertRaises(TypeError):
            plot_utils.rainbow_plot(notAxis, np.cos(t_vec) * np.linspace(0, 1, num_pts),
                                     np.sin(t_vec) * np.linspace(0, 1, num_pts),
                                     num_steps=32)

    """
    def test_xvec_not_array(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(TypeError):
            plot_utils.rainbow_plot(axis, 'hello',
                                     np.sin(t_vec) * np.linspace(0, 1, num_pts),
                                     num_steps=32)

    def test_yvec_not_a1darrray(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(AssertionError):
            plot_utils.rainbow_plot(axis, np.cos(t_vec) * np.linspace(0, 1, num_pts),
                                    np.arange(100).reshape(10,10), num_steps=32)

    def test_xvec_not_a1darrray(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(AssertionError):
            plot_utils.rainbow_plot(axis, np.arange(100).reshape(10,10),
                                     np.cos(t_vec) * np.linspace(0, 1, num_pts), num_steps=32)

    def test_yvec_not_same_xvec(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(ValueError):
            plot_utils.rainbow_plot(axis, np.cos(t_vec) * np.linspace(0, 1, num_pts-1),
                                     np.sin(t_vec) * np.linspace(0, 1, num_pts), num_steps=32)

    def test_num_steps_not_num(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(TypeError):
            plot_utils.rainbow_plot(axis, np.cos(t_vec) * np.linspace(0, 1, num_pts),
                                     np.sin(t_vec) * np.linspace(0, 1, num_pts),
                                     num_steps='hello')

    def test_base(self):
        num_pts = 1024
        t_vec = np.linspace(0, 10 * np.pi, num_pts)

        fig, axis = plt.subplots(figsize=(4, 4))
        plot_utils.rainbow_plot(axis, np.cos(t_vec) * np.linspace(0, 1, num_pts),
                                     np.sin(t_vec) * np.linspace(0, 1, num_pts),
                                     num_steps=32)
    """


class TestPlotLineFamily(unittest.TestCase):

    """
    def test_base(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = range(1, 5)
        y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])
        freq_strs = [str(_) for _ in freqs]

        fig, axis = plt.subplots(figsize=(12, 4))
        plot_utils.plot_line_family(axis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)
    """

    def test_plot_line_family_not_axis(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = range(1, 5)
        y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])
        freq_strs = [str(_) for _ in freqs]
        notAxis = 'hello'
        with self.assertRaises(TypeError):
            plot_utils.plot_line_family(notAxis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)

    """
    def test_plot_line_family_not_xvec(self):
        x_vec = 'hello'
        freqs = range(1, 5)
        y_mat = np.array([freq for freq in freqs])
        freq_strs = [str(_) for _ in freqs]

        fig, axis = plt.subplots(figsize=(12, 4))
        with self.assertRaises(TypeError):
            plot_utils.plot_line_family(axis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)

    def test_plot_line_family_not_ymat(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = range(1, 5)
        y_mat = np.zeros_like(x_vec)
        freq_strs = [str(_) for _ in freqs]
        fig, axis = plt.subplots(ncols=2, figsize=(12, 4))
        with self.assertRaises(TypeError):
            plot_utils.plot_line_family(axis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)

    def test_plot_line_family_not_freqstrs(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = range(1, 5)
        y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])
        freq_strs = 5

        fig, axis = plt.subplots(figsize=(12, 4))
        with self.assertRaises(TypeError):
            plot_utils.plot_line_family(axis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)

    def test_plot_line_family_not_labelprefix(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = range(1, 5)
        y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])
        freq_strs = [str(_) for _ in freqs]

        fig, axis = plt.subplots(figsize=(12, 4))
        with self.assertRaises(TypeError):
            plot_utils.plot_line_family(axis, x_vec, y_mat,
                               line_names=freq_strs, label_prefix= 6, label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)
    """

class TestPlotMap(unittest.TestCase):

    pass
    """
    def test_plot_map(self):
        x_vec = np.linspace(0, 6 * np.pi, 256)
        y_vec = np.sin(x_vec) ** 2
    
        atom_intensities = y_vec * np.atleast_2d(y_vec).T
    
        fig, axis = plt.subplots()
        plot_utils.plot_map(axis, atom_intensities, stdevs=1.5, num_ticks=4,
                            x_vec=np.linspace(-1, 1, atom_intensities.shape[0]),
                            y_vec=np.linspace(0, 500, atom_intensities.shape[1]),
                            cbar_label='intensity (a. u.)', tick_font_size=16)
    
    """


class TestPlotCurves(unittest.TestCase):

    pass
    """
    def test_plot_curves(self):
        x_vec = np.linspace(0, 2 * np.pi, 256)
        freqs = np.linspace(0.5, 5, 9)
        y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])

        plot_utils.plot_curves(x_vec, y_mat)
    """


class TestPlotComplexSpectra(unittest.TestCase):

    @staticmethod
    def get_complex_2d_image(freq):
        # Simple function to generate images
        x_vec = np.linspace(0, freq * np.pi, 256)
        y_vec_1 = np.sin(x_vec) ** 2
        y_vec_2 = np.cos(x_vec) ** 2
        return y_vec_2 * np.atleast_2d(y_vec_2).T + 1j * (y_vec_1 * np.atleast_2d(y_vec_1).T)
    
    """
    def test_plot_complex_spectra(self):
        # The range of frequences over which the images are generated
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        plot_utils.plot_complex_spectra(np.array(image_stack))
    """

    def test_not_map_stack(self):
        with self.assertRaises(TypeError):
            plot_utils.plot_complex_spectra('wrongthing')

    def test_not_x_vec(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(TypeError):
            plot_utils.plot_complex_spectra(np.array(image_stack), x_vec='notvec')

    def test_is_2d_x_vec(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(ValueError):
            plot_utils.plot_complex_spectra(np.array(image_stack), [[1]])

    def test_is_not_dim_x_vec(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(ValueError):
            plot_utils.plot_complex_spectra(np.array(image_stack), [1])

    def test_is_x_vec(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        ran_arr = np.zeros_like(image_stack)
        with self.assertRaises(ValueError):
            plot_utils.plot_complex_spectra(np.array(image_stack), ran_arr)

    """
    def test_num_comps(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [TestPlotFeatures.get_complex_2d_image(freq) for freq in frequencies]
        plot_utils.plot_complex_spectra(np.array(image_stack), num_comps=None)
    """

    def test_num_comps_not_int(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(TypeError):
            plot_utils.plot_complex_spectra(np.array(image_stack), num_comps='wrong')

    def test_not_str(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(TypeError):
            plot_utils.plot_complex_spectra(np.array(image_stack), title=1)

    def test_not_stdevs(self):
        frequencies = 2 ** np.arange(4)
        image_stack = [self.get_complex_2d_image(freq) for freq in frequencies]
        with self.assertRaises(TypeError):
            plot_utils.plot_complex_spectra(np.array(image_stack), stdevs=-1)


class TestPlotScree(unittest.TestCase):

    """
    def test_simple(self):
        scree = np.exp(-1 * np.arange(100))
        plot_utils.plot_scree(scree, color='r')
    """

    def test_title_wrong(self):
        scree = np.exp(-1 * np.arange(100))
        with self.assertRaises(TypeError):
            plot_utils.plot_scree(scree, title=1)

    def test_scree_wrong(self):
        scree = 'string'
        with self.assertRaises(TypeError):
            plot_utils.plot_scree(scree)

    """
    def test_scree_list(self):
        scree = np.arange(5)
        plot_utils.plot_scree(scree, color='r')

    def get_sine_2d_image(freq):
        x_vec = np.linspace(0, freq*np.pi, 256)
        y_vec = np.sin(x_vec)**2
        return y_vec * np.atleast_2d(y_vec).T
    """


class TestMapStack(unittest.TestCase):

    pass
    """
    def test_map_stack(self):
        def get_sine_2d_image(freq):
            x_vec = np.linspace(0, freq*np.pi, 256)
            y_vec = np.sin(x_vec)**2
            return y_vec * np.atleast_2d(y_vec).T
        frequencies = [0.25, 0.5, 1, 2, 4 ,8, 16, 32, 64]
        image_stack = [get_sine_2d_image(freq) for freq in frequencies]
        image_stack = np.array(image_stack)
        fig, axes = plot_utils.plot_map_stack(image_stack, reverse_dims=False, title_yoffset=0.95)
    """


class TestCbarForLinePlot(unittest.TestCase):

    def test_not_axis(self):
        with self.assertRaises(TypeError):
            plot_utils.cbar_for_line_plot(1, 2)

    """
    def test_neg_num_steps(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(ValueError):
            plot_utils.cbar_for_line_plot(axis, -2)

    def test_not_int_num_steps(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(TypeError):
            plot_utils.cbar_for_line_plot(axis, 'hello')

    def test_ticks_not_boolean(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        with self.assertRaises(AssertionError):
            plot_utils.cbar_for_line_plot(axis, 2, discrete_ticks='hello')

    def test_complete_func(self):
        fig, axis = plt.subplots(figsize=(4, 4))
        plot_utils.cbar_for_line_plot(axis, 2)
    """


if __name__ == '__main__':
    unittest.main()
