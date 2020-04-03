from __future__ import division, print_function, unicode_literals, absolute_import
import os
import sys
import socket
from warnings import warn
import h5py
import numpy as np
from io import StringIO
from contextlib import contextmanager
from pyUSID import __version__
from platform import platform

sys.path.append("../../pyUSID/")
from pyUSID.io.hdf_utils import get_attr
from pyUSID.io.io_utils import get_time_stamp
from pyUSID.io.write_utils import INDICES_DTYPE, VALUES_DTYPE

std_beps_path = 'test_hdf_utils.h5'
sparse_sampling_path = 'sparse_sampling.h5'
incomplete_measurement_path = 'incomplete_measurement.h5'
relaxation_path = 'relaxation.h5'

sys.path.append("../../pyUSID/")
from pyUSID.io.write_utils import build_ind_val_matrices

if sys.version_info.major == 3:
    unicode = str


def delete_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_safe_attrs(h5_object, attrs):
    for key, val in attrs.items():
        h5_object.attrs[key] = val


def write_string_list_as_attr(h5_object, attrs):
    for key, val in attrs.items():
        h5_object.attrs[key] = np.array(val, dtype='S')


def write_aux_reg_ref(h5_dset, labels, is_spec=True):
    for index, reg_ref_name in enumerate(labels):
        if is_spec:
            reg_ref_tuple = (slice(index, index + 1), slice(None))
        else:
            reg_ref_tuple = (slice(None), slice(index, index + 1))
        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]


def write_main_reg_refs(h5_dset, attrs):
    for reg_ref_name, reg_ref_tuple in attrs.items():
        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]
    write_string_list_as_attr(h5_dset, {'labels': list(attrs.keys())})


@contextmanager
def capture_stdout():
    """
    context manager encapsulating a pattern for capturing stdout writes
    and restoring sys.stdout even upon exceptions

    https://stackoverflow.com/questions/17067560/intercept-pythons-print-statement-and-display-in-gui

    Examples:
    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     captured = get_value()
    >>> print('Gotcha: ' + captured)

    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     raise Exception('oh no!')
    >>> print('Does printing still work?')
    """
    # Redirect sys.stdout
    out = StringIO()
    sys.stdout = out
    # Yield a method clients can use to obtain the value
    try:
        yield out.getvalue
    finally:
        # Restore the normal stdout
        sys.stdout = sys.__stdout__


def validate_aux_dset_pair(test_class, h5_group, h5_inds, h5_vals, dim_names, dim_units, inds_matrix,
                           vals_matrix=None, base_name=None, h5_main=None, is_spectral=True,
                           slow_to_fast=False):
    if vals_matrix is None:
        vals_matrix = inds_matrix
    if base_name is None:
        if is_spectral:
            base_name = 'Spectroscopic'
        else:
            base_name = 'Position'
    else:
        test_class.assertIsInstance(base_name, (str, unicode))

    if not slow_to_fast:
        # Sending in to Fast to Slow but what comes out is slow to fast
        func = np.flipud if is_spectral else np.fliplr

        print(inds_matrix)

        vals_matrix = func(vals_matrix)
        inds_matrix = func(inds_matrix)

        dim_names = dim_names[::-1]
        dim_units = dim_units[::-1]

    for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_inds, h5_vals],
                                                      [INDICES_DTYPE, VALUES_DTYPE],
                                                      [base_name + '_Indices', base_name + '_Values'],
                                                      [inds_matrix, vals_matrix]):
        if isinstance(h5_main, h5py.Dataset):
            test_class.assertEqual(h5_main.file[h5_main.attrs[exp_name]], h5_dset)
        test_class.assertIsInstance(h5_dset, h5py.Dataset)
        test_class.assertEqual(h5_dset.parent, h5_group)
        test_class.assertEqual(h5_dset.name.split('/')[-1], exp_name)
        test_class.assertTrue(np.allclose(ref_data, h5_dset[()]))
        test_class.assertEqual(h5_dset.dtype, exp_dtype)
        test_class.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
        test_class.assertTrue(np.all([x == y for x, y in zip(dim_names, get_attr(h5_dset, 'labels'))]))
        test_class.assertTrue(np.all([x == y for x, y in zip(dim_units, get_attr(h5_dset, 'units'))]))
        # assert region references even though these are not really used anywhere
        for dim_ind, curr_name in enumerate(dim_names):
            if is_spectral:
                expected = np.squeeze(ref_data[dim_ind])
            else:
                expected = np.squeeze(ref_data[:, dim_ind])
            actual = np.squeeze(h5_dset[h5_dset.attrs[curr_name]])
            try:
                match = np.allclose(expected, actual)
            except ValueError:
                match = False
            if match:
                test_class.assertTrue(match)
            else:
                warn('Test for region reference: ' + curr_name + ' failed')


def verify_book_keeping_attrs(test_class, h5_obj):
    time_stamp = get_time_stamp()
    in_file = h5_obj.attrs['timestamp']
    test_class.assertEqual(time_stamp[:time_stamp.rindex('_')], in_file[:in_file.rindex('_')])
    test_class.assertEqual(__version__, h5_obj.attrs['pyUSID_version'])
    test_class.assertEqual(socket.getfqdn(), h5_obj.attrs['machine_id'])
    test_class.assertEqual(platform(), h5_obj.attrs['platform'])


def make_sparse_sampling_file():
    if os.path.exists(sparse_sampling_path):
        os.remove(sparse_sampling_path)

    h5_main = None

    with h5py.File(sparse_sampling_path, mode='w') as h5_f:
        h5_meas_grp = h5_f.create_group('Measurement_000')

        freq_pts = 3
        spec_inds = np.expand_dims(np.arange(freq_pts), 0)
        spec_vals = np.expand_dims(np.linspace(300, 330, freq_pts), 0)

        h5_spec_inds = h5_meas_grp.create_dataset('Spectroscopic_Indices', data=spec_inds, dtype=np.uint16)
        h5_spec_vals = h5_meas_grp.create_dataset('Spectroscopic_Values', data=spec_vals, dtype=np.float32)

        spec_attrs = {'labels': ['Frequency'], 'units': ['kHz']}

        for dset in [h5_spec_inds, h5_spec_vals]:
            write_aux_reg_ref(dset, spec_attrs['labels'], is_spec=True)
            write_string_list_as_attr(dset, spec_attrs)

        import random

        num_rows = 5
        num_cols = 7

        full_vals = np.vstack((np.tile(np.arange(num_cols), num_rows),
                               np.repeat(np.arange(num_rows), num_cols))).T
        full_vals = np.vstack((full_vals[:, 0] * 50, full_vals[:, 1] * 1.25)).T

        fract = 0.25
        chosen_pos = random.sample(range(num_rows * num_cols), int(fract * num_rows * num_cols))

        pos_inds = np.tile(np.arange(len(chosen_pos)), (2, 1)).T
        pos_vals = full_vals[chosen_pos]

        pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

        h5_chan_grp_1 = h5_meas_grp.create_group('Channel_000')
        h5_chan_grp_2 = h5_meas_grp.create_group('Channel_001')

        for h5_chan_grp, add_attribute in zip([h5_chan_grp_1, h5_chan_grp_2], [False, True]):

            this_pos_attrs = pos_attrs.copy()
            if add_attribute:
                this_pos_attrs.update({'incomplete_dimensions': ['X', 'Y']})

            h5_pos_inds = h5_chan_grp.create_dataset('Position_Indices', data=pos_inds, dtype=np.uint16)
            h5_pos_vals = h5_chan_grp.create_dataset('Position_Values', data=pos_vals, dtype=np.float32)

            for dset in [h5_pos_inds, h5_pos_vals]:
                write_aux_reg_ref(dset, this_pos_attrs['labels'], is_spec=False)
                write_string_list_as_attr(dset, this_pos_attrs)

            h5_main = h5_chan_grp.create_dataset('Raw_Data',
                                                 data=np.random.rand(len(chosen_pos), freq_pts),
                                                 dtype=np.float32)

            # Write mandatory attributes:
            write_safe_attrs(h5_main, {'units': 'V', 'quantity': 'Cantilever Deflection'})

            # Link ancillary
            for dset in [h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals]:
                h5_main.attrs[dset.name.split('/')[-1]] = dset.ref
    return h5_meas_grp

def make_incomplete_measurement_file():
    if os.path.exists(incomplete_measurement_path):
        os.remove(incomplete_measurement_path)

    with h5py.File(incomplete_measurement_path, mode='w') as h5_f:
        h5_meas_grp = h5_f.create_group('Measurement_000')

        freq_pts = 3
        dc_offsets = 2

        spec_inds = np.vstack((np.tile(np.arange(freq_pts), dc_offsets),
                               np.repeat(np.arange(dc_offsets), freq_pts)))

        # make the values more interesting:
        spec_vals = np.vstack((np.tile(np.linspace(320, 340, freq_pts), dc_offsets),
                               np.repeat(np.linspace(-8, 8, dc_offsets), freq_pts)))

        h5_spec_inds = h5_meas_grp.create_dataset('Spectroscopic_Indices', data=spec_inds, dtype=np.uint16)
        h5_spec_vals = h5_meas_grp.create_dataset('Spectroscopic_Values', data=spec_vals, dtype=np.float32)

        spec_attrs = {'labels': ['Frequency', 'DC_Offset'], 'units': ['kHz', 'V']}

        for dset in [h5_spec_inds, h5_spec_vals]:
            write_aux_reg_ref(dset, spec_attrs['labels'], is_spec=True)
            write_string_list_as_attr(dset, spec_attrs)

        num_rows = 5
        num_cols = 7

        pos_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        # make the values more interesting:
        pos_vals = np.vstack((pos_inds[:, 0] * 50, pos_inds[:, 1] * 1.25)).T

        pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

        tot_positions = 4 * num_cols + 3
        incomp_dim_names = ['X', 'Y']

        def _create_pos_and_main(h5_group, add_attribute):
            h5_pos_inds = h5_group.create_dataset('Position_Indices',
                                                  data=pos_inds[:tot_positions],
                                                  dtype=np.uint16)
            h5_pos_vals = h5_group.create_dataset('Position_Values',
                                                  data=pos_vals[:tot_positions],
                                                  dtype=np.float32)

            this_pos_attrs = pos_attrs.copy()
            if add_attribute:
                this_pos_attrs.update({'incomplete_dimensions': incomp_dim_names})

            for dset in [h5_pos_inds, h5_pos_vals]:
                write_aux_reg_ref(dset, pos_attrs['labels'], is_spec=False)
                write_string_list_as_attr(dset, this_pos_attrs)

            h5_main = h5_group.create_dataset('Raw_Data',
                                              data=np.random.rand(tot_positions, freq_pts * dc_offsets),
                                              dtype=np.float32)

            # Write mandatory attributes:
            write_safe_attrs(h5_main, {'units': 'V', 'quantity': 'Cantilever Deflection'})

            # Link ancillary    
            for dset in [h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals]:
                h5_main.attrs[dset.name.split('/')[-1]] = dset.ref

            return h5_main

        h5_main_1 = _create_pos_and_main(h5_meas_grp.create_group('Channel_000'), False)

        h5_main_2 = _create_pos_and_main(h5_meas_grp.create_group('Channel_001'), True)


def make_relaxation_file():
    if os.path.exists(relaxation_path):
        os.remove(relaxation_path)

    with h5py.File(relaxation_path, mode='w') as h5_f:
        h5_meas_grp = h5_f.create_group('Measurement_000')

        num_rows = 2
        num_cols = 11

        pos_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        # make the values more interesting:
        pos_vals = np.vstack((pos_inds[:, 0] * 50, pos_inds[:, 1] * 1.25)).T

        pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

        h5_pos_inds = h5_meas_grp.create_dataset('Position_Indices', data=pos_inds, dtype=np.uint16)
        h5_pos_vals = h5_meas_grp.create_dataset('Position_Values', data=pos_vals, dtype=np.float32)

        for dset in [h5_pos_inds, h5_pos_vals]:
            write_aux_reg_ref(dset, pos_attrs['labels'], is_spec=False)
            write_string_list_as_attr(dset, pos_attrs)

        spec_attrs = {'labels': ['Frequency', 'Repeats', 'DC_Offset', 'Field'],
                      'units': ['kHz', 'a. u.', 'V', 'a.u.']}

        freq_pts = 3
        repeats = 5
        dc_offsets = 7
        field_inds = 1

        spec_unit_vals = [np.linspace(320, 340, freq_pts),
                          np.arange(repeats),
                          3 * np.pi * np.linspace(0, 1, dc_offsets),
                          np.array([1, 0])]

        spec_ind_mat, spec_val_mat = build_ind_val_matrices(spec_unit_vals[:-1])

        # Manually creating the field array that starts with 1
        field_ind_unit = np.hstack(([0], np.ones(repeats - field_inds, dtype=np.uint16)))
        field_val_unit = np.hstack(([1], np.zeros(repeats - field_inds, dtype=np.uint16)))

        # Manually appending to the indices and values table
        spec_ind_mat = np.vstack((spec_ind_mat,
                                  np.tile(np.repeat(field_ind_unit, freq_pts), dc_offsets)))

        spec_val_mat = np.vstack((spec_val_mat,
                                  np.tile(np.repeat(field_val_unit, freq_pts), dc_offsets)))

        spec_unit_vals_dict = dict()
        for dim_ind, dim_unit_vals in enumerate(spec_unit_vals):
            spec_unit_vals_dict['unit_vals_dim_' + str(dim_ind)] = dim_unit_vals

        h5_chan_grp_1 = h5_meas_grp.create_group('Channel_000')
        h5_chan_grp_2 = h5_meas_grp.create_group('Channel_001')

        for h5_chan_grp, add_attribute in zip([h5_chan_grp_1, h5_chan_grp_2], [False, True]):

            h5_spec_inds = h5_chan_grp.create_dataset('Spectroscopic_Indices', data=spec_ind_mat, dtype=np.uint16)
            h5_spec_vals = h5_chan_grp.create_dataset('Spectroscopic_Values', data=spec_val_mat, dtype=np.float32)

            this_spec_attrs = spec_attrs.copy()
            if add_attribute:
                this_spec_attrs.update({'dependent_dimensions': ['Field']})

            for dset in [h5_spec_inds, h5_spec_vals]:
                write_aux_reg_ref(dset, spec_attrs['labels'], is_spec=True)
                write_string_list_as_attr(dset, this_spec_attrs)
                # Write the unit values as attributes - testing purposes only:
                write_safe_attrs(dset, spec_unit_vals_dict)

            h5_main = h5_chan_grp.create_dataset('Raw_Data',
                                                 data=np.random.rand(num_rows * num_cols,
                                                                     freq_pts * repeats * dc_offsets),
                                                 dtype=np.float32)

            # Write mandatory attributes:
            write_safe_attrs(h5_main, {'units': 'V', 'quantity': 'Cantilever Deflection'})

            # Link ancillary
            for dset in [h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals]:
                h5_main.attrs[dset.name.split('/')[-1]] = dset.ref


def make_beps_file(rev_spec=False):
    if os.path.exists(std_beps_path):
        os.remove(std_beps_path)

    with h5py.File(std_beps_path, mode='w') as h5_f:

        h5_raw_grp = h5_f.create_group('Raw_Measurement')
        write_safe_attrs(h5_raw_grp, {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
        write_string_list_as_attr(h5_raw_grp, {'att_4': ['str_1', 'str_2', 'str_3']})

        _ = h5_raw_grp.create_group('Misc')

        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        source_dset_name = 'source_main'
        tool_name = 'Fitter'

        # Per USID, dimensions are arranged from fastest to slowest
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

        h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
        write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
        write_string_list_as_attr(h5_pos_inds, pos_attrs)

        # make the values more interesting:
        cols_offset = -750
        cols_step = 50
        rows_offset = 2
        rows_step = 1.25
        source_pos_data = np.vstack((cols_offset + source_pos_data[:, 0] * cols_step,
                                     rows_offset + source_pos_data[:, 1] * rows_step)).T

        _ = h5_raw_grp.create_dataset('X', data=cols_offset + cols_step * np.arange(num_cols))
        _ = h5_raw_grp.create_dataset('Y', data=rows_offset + rows_step * np.arange(num_rows))

        h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
        write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
        write_string_list_as_attr(h5_pos_vals, pos_attrs)

        if rev_spec:
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(np.arange(num_cycle_pts), num_cycles)))
            source_spec_attrs = {'units': ['', 'V'],
                                 'labels': ['Cycle', 'Bias']}
        else:
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

        h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                        dtype=np.uint16)
        write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

        # make spectroscopic axis interesting as well
        bias_amp = 2.5
        bias_period = np.pi
        bias_vec = bias_amp * np.sin(np.linspace(0, bias_period, num_cycle_pts, endpoint=False))

        _ = h5_raw_grp.create_dataset('Bias', data=bias_vec)
        _ = h5_raw_grp.create_dataset('Cycle', data=np.arange(num_cycles))

        if rev_spec:
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(bias_vec, num_cycles)))
        else:
            source_spec_data = np.vstack((np.tile(bias_vec, num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))

        h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                        dtype=np.float32)
        write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

        main_nd = np.random.rand(num_rows, num_cols, num_cycles, num_cycle_pts)
        h5_nd_main = h5_raw_grp.create_dataset('n_dim_form', data=main_nd)
        write_string_list_as_attr(h5_nd_main, {'dims': ['Y', 'X', 'Cycle', 'Bias']})

        if rev_spec:
            # This simulates things like BEPS where Field should actually be varied slower but is varied faster during acquisition
            main_nd = main_nd.transpose(0, 1, 3, 2)

        source_main_data = main_nd.reshape(num_rows * num_cols, num_cycle_pts * num_cycles)
        # source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
        h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
        write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})
        write_main_reg_refs(h5_source_main, {'even_rows': (slice(0, None, 2), slice(None)),
                                                            'odd_rows': (slice(1, None, 2), slice(None))})

        # Now need to link as main!
        for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
            h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

        _ = h5_raw_grp.create_dataset('Ancillary', data=np.arange(5))

        # Now add a few results:

        h5_results_grp_1 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_000')
        write_safe_attrs(h5_results_grp_1,
                                        {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
        write_string_list_as_attr(h5_results_grp_1, {'att_4': ['str_1', 'str_2', 'str_3']})

        num_cycles = 1
        num_cycle_pts = 7

        results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
        results_spec_attrs = {'units': ['V'], 'labels': ['Bias']}

        h5_results_1_spec_inds = h5_results_grp_1.create_dataset('Spectroscopic_Indices',
                                                                 data=results_spec_inds, dtype=np.uint16)
        write_aux_reg_ref(h5_results_1_spec_inds, results_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_results_1_spec_inds, results_spec_attrs)

        results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)

        h5_results_1_spec_vals = h5_results_grp_1.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                 dtype=np.float32)
        write_aux_reg_ref(h5_results_1_spec_vals, results_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_results_1_spec_vals, results_spec_attrs)

        # Let this be a compound dataset:
        struc_dtype = np.dtype({'names': ['r', 'g', 'b'],
                                'formats': [np.float32, np.float16,
                                            np.float64]})

        num_elems = (num_rows, num_cols, num_cycles, num_cycle_pts)
        results_1_nd = np.zeros(shape=num_elems, dtype=struc_dtype)
        for name_ind, name in enumerate(struc_dtype.names):
            results_1_nd[name] = np.random.random(size=num_elems)

        h5_results_1_nd = h5_results_grp_1.create_dataset('n_dim_form',
                                                          data=results_1_nd)
        write_string_list_as_attr(h5_results_1_nd,
                                  {'dims': ['Y', 'X', 'Cycle', 'Bias']})

        results_1_main_data = results_1_nd.reshape(num_rows * num_cols,
                                                   num_cycle_pts * num_cycles)

        h5_results_1_main = h5_results_grp_1.create_dataset('results_main', data=results_1_main_data)
        write_safe_attrs(h5_results_1_main, {'units': 'pF', 'quantity': 'Capacitance'})

        # Now need to link as main!
        for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
            h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

        # add another result with different parameters

        h5_results_grp_2 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_001')
        write_safe_attrs(h5_results_grp_2,
                                        {'att_1': 'other_string_val', 'att_2': 5.4321, 'att_3': [4, 1, 3]})
        write_string_list_as_attr(h5_results_grp_2, {'att_4': ['s', 'str_2', 'str_3']})

        # Let these results be a complex typed dataset:
        results_2_nd = np.random.random(size=num_elems) + \
                       1j * np.random.random(size=num_elems)

        h5_results_2_nd = h5_results_grp_2.create_dataset('n_dim_form',
                                                          data=results_2_nd)
        write_string_list_as_attr(h5_results_2_nd,
                                  {'dims': ['Y', 'X', 'Cycle', 'Bias']})

        results_2_main_data = results_2_nd.reshape(num_rows * num_cols,
                                                   num_cycle_pts * num_cycles)
        h5_results_2_main = h5_results_grp_2.create_dataset('results_main', data=results_2_main_data)
        write_safe_attrs(h5_results_2_main, {'units': 'pF', 'quantity': 'Capacitance'})

        h5_results_2_spec_inds = h5_results_grp_2.create_dataset('Spectroscopic_Indices',
                                                                 data=results_spec_inds, dtype=np.uint16)
        write_aux_reg_ref(h5_results_2_spec_inds, results_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_results_2_spec_inds, results_spec_attrs)

        h5_results_2_spec_vals = h5_results_grp_2.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                 dtype=np.float32)
        write_aux_reg_ref(h5_results_2_spec_vals, results_spec_attrs['labels'], is_spec=True)
        write_string_list_as_attr(h5_results_2_spec_vals, results_spec_attrs)

        # Now need to link as main!
        for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
            h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref
