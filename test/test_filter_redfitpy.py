"""
pytest for filter/redfitpy.py
"""

import pytest

import easyclimate as ecl
import numpy as np
import xarray as xr
from pathlib import Path
from .const_define import TEST_DATA_PATH

sst_data = xr.open_dataset(str(Path(TEST_DATA_PATH, "test_input_nino34.nc"))).sst


def test_calc_redfit1():
    result_data = ecl.filter.calc_redfit(sst_data)

    result_data1 = result_data.gxx[20:30].data
    result_data3 = result_data.gred_th[20:30].data
    result_data6 = result_data.chi2_80[20:30].data
    result_data7 = result_data.chi2_90[20:30].data
    result_data8 = result_data.chi2_95[20:30].data
    result_data9 = result_data.chi2_99[20:30].data

    refer_data1 = np.array(
        [
            2.6100595,
            1.3072749,
            7.6913514,
            1.6619556,
            5.2785249,
            3.2057655,
            0.66721755,
            17.080835,
            15.754893,
            4.1825418,
        ]
    )
    refer_data3 = np.array(
        [
            4.3748851,
            3.9981205,
            3.667268,
            3.3753133,
            3.1165028,
            2.886091,
            2.6801355,
            2.4953432,
            2.3289518,
            2.1786287,
        ]
    )
    refer_data6 = np.array(
        [
            7.0426998,
            6.4361835,
            5.9035764,
            5.4335871,
            5.016953,
            4.6460357,
            4.3144879,
            4.0170093,
            3.7491519,
            3.5071614,
        ]
    )
    refer_data7 = np.array(
        [
            10.071729,
            9.2043524,
            8.4426737,
            7.7705445,
            7.1747184,
            6.6442714,
            6.1701269,
            5.7447038,
            5.3616424,
            5.015573,
        ]
    )
    refer_data8 = np.array(
        [
            13.109101,
            11.980147,
            10.988766,
            10.113939,
            9.3384285,
            8.6480122,
            8.0308781,
            7.4771581,
            6.9785752,
            6.5281405,
        ]
    )
    refer_data9 = np.array(
        [
            20.143457,
            18.408705,
            16.885347,
            15.541089,
            14.349437,
            13.288543,
            12.340254,
            11.489408,
            10.723285,
            10.031146,
        ]
    )

    assert np.isclose(result_data1, refer_data1, atol=0.01).all()
    assert np.isclose(result_data3, refer_data3, atol=0.01).all()
    assert np.isclose(result_data6, refer_data6, atol=0.01).all()
    assert np.isclose(result_data7, refer_data7, atol=0.01).all()
    assert np.isclose(result_data8, refer_data8, atol=0.01).all()
    assert np.isclose(result_data9, refer_data9, atol=0.01).all()


def test_calc_redfit2():
    result_data = ecl.filter.calc_redfit(sst_data, iwin="welch")
    result_data1 = result_data.gxx[20:30].data
    refer_data1 = np.array(
        [
            1.1525935,
            1.5818458,
            6.1552815,
            5.9970174,
            4.5128694,
            3.7235408,
            0.19082901,
            21.362429,
            19.166662,
            2.2907004,
        ]
    )
    assert np.isclose(result_data1, refer_data1, atol=0.01).all()


def test_calc_redfit3():
    result_data = ecl.filter.calc_redfit(sst_data, iwin="hanning")
    result_data1 = result_data.gxx[20:30].data
    refer_data1 = np.array(
        [
            0.48377445,
            1.1684574,
            5.5684972,
            7.5657225,
            6.037571,
            3.4565904,
            1.446534,
            21.131706,
            16.222567,
            0.60975897,
        ]
    )
    assert np.isclose(result_data1, refer_data1, atol=0.01).all()


def test_calc_redfit4():
    result_data = ecl.filter.calc_redfit(sst_data, iwin="triangular")
    result_data1 = result_data.gxx[20:30].data
    refer_data1 = np.array(
        [
            0.53083158,
            1.0754435,
            5.9494491,
            6.8237329,
            5.0147862,
            3.5979056,
            0.74510956,
            19.944063,
            16.900396,
            1.1186559,
        ]
    )
    assert np.isclose(result_data1, refer_data1, atol=0.01).all()


def test_calc_redfit5():
    result_data = ecl.filter.calc_redfit(sst_data, iwin="blackmanharris")
    result_data1 = result_data.gxx[20:30].data
    refer_data1 = np.array(
        [
            0.25909352,
            0.9457823,
            5.4999166,
            8.3161945,
            6.9438009,
            3.0793602,
            2.3809063,
            19.174746,
            13.924948,
            0.18201128,
        ]
    )
    assert np.isclose(result_data1, refer_data1, atol=0.01).all()


def test_calc_redfit6():
    result_data = ecl.filter.calc_redfit(sst_data, iwin="rectangular", mctest=True)
    result_data1 = result_data.gxx[20:30].data
    refer_data1 = np.array(
        [
            2.6100595,
            1.3072749,
            7.6913514,
            1.6619556,
            5.2785249,
            3.2057655,
            0.66721755,
            17.080835,
            15.754893,
            4.1825418,
        ]
    )
    assert np.isclose(result_data1, refer_data1, atol=0.01).all()


def test_calc_redfit_cross1():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20]
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.49393216, 0.043295149, 0.10684578, 0.036448568, 0.076293454, 0.0013848162]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_calc_redfit_cross2():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20], iwin="welch"
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.15475142, 0.062119752, 0.026066253, 0.025041819, 0.024693901, 0.024360122]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_calc_redfit_cross3():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20], iwin="hanning"
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.10175402, 0.064516999, 0.017830357, 0.0038753576, 0.048153084, 0.0099583864]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_calc_redfit_cross4():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20], iwin="triangular"
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.088216975, 0.061784219, 0.013301093, 0.024668887, 0.028980276, 0.024734346]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_calc_redfit_cross5():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20], iwin="blackmanharris"
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.064234719, 0.061255176, 0.021671789, 0.0083046546, 0.038331024, 0.0137163]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_calc_redfit_cross6():
    gxx, gyy, gxy, cxy, phxy = ecl.filter.calc_redfit_cross(
        sst_data.data[:20], sst_data.data[:20], mctest=False, mctest_phi=False
    )
    result_gxx_data1 = gxx.gxx[5:].data
    refer_gxx_data1 = np.array(
        [0.49393216, 0.043295149, 0.10684578, 0.036448568, 0.076293454, 0.0013848162]
    )

    assert np.isclose(result_gxx_data1, refer_gxx_data1, atol=0.01).all()


def test_redfit_monte_carlo_is_repeatable():
    first = ecl.filter.calc_redfit(sst_data[:61], nsim=21, mctest=True)
    second = ecl.filter.calc_redfit(sst_data[:61], nsim=21, mctest=True)
    xr.testing.assert_identical(first.gred, second.gred)
    xr.testing.assert_identical(first.ci90, second.ci90)


def test_redfit_cross_monte_carlo_is_repeatable():
    # Unequal lengths and an odd number of simulations leave a cached Gaussian draw.
    first, *_ = ecl.filter.calc_redfit_cross(
        sst_data.data[:21], sst_data.data[:20], nsim=21, mctest_phi=False
    )
    second, *_ = ecl.filter.calc_redfit_cross(
        sst_data.data[:21], sst_data.data[:20], nsim=21, mctest_phi=False
    )
    xr.testing.assert_identical(first.gred, second.gred)
    xr.testing.assert_identical(first.ci90, second.ci90)
