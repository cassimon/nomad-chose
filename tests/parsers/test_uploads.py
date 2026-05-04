import os.path

import pytest
from nomad.client import normalize_all, parse
from nomad.metainfo.metainfo import MProxy

from nomad_chose.schema_packages.schema_package import (
    LabEQEMeasurement,
    LabJVMeasurement,
    LabStabilityMeasurement,
)


def test_upload_jv_stability_sidecar_parses():
    """Sidecar YAML loads the correct schema class with fields set from YAML."""
    test_file = os.path.join('tests', 'data', '02_jv_forward.archive.yaml')
    entry_archive = parse(test_file)[0]

    assert isinstance(entry_archive.data, LabJVMeasurement)
    assert entry_archive.data.jv_file.endswith('(JV)_AI03-1A.txt')
    assert entry_archive.data.operator == 'Alice'
    # pvk_sample reference is stored as an unresolved proxy (resolved only at upload time)
    assert entry_archive.data.pvk_sample is not None


def test_upload_stability_sidecar_parses():
    """Sidecar YAML for combined stability loads both file names."""
    test_file = os.path.join('tests', 'data', '03_jv_reverse.archive.yaml')
    entry_archive = parse(test_file)[0]

    assert isinstance(entry_archive.data, LabStabilityMeasurement)
    assert entry_archive.data.stability_parameters_file.endswith('(Parameters)_AI03-1A.txt')
    assert entry_archive.data.stability_tracking_file.endswith('(Tracking)_AI03-1A.txt')
    assert entry_archive.data.operator == 'Alice'
    assert entry_archive.data.pvk_sample is not None


def test_upload_ipce_sidecar_parses():
    """Sidecar YAML for IPCE/EQE loads correct file name."""
    test_file = os.path.join('tests', 'data', '04_eqe_archive.yaml')
    entry_archive = parse(test_file)[0]

    assert isinstance(entry_archive.data, LabEQEMeasurement)
    assert entry_archive.data.eqe_file.endswith('_IPCE_AI03.txt')
    assert entry_archive.data.operator == 'Bob'
    assert entry_archive.data.pvk_sample is not None


def test_upload_stability_tracking_sidecar_parses():
    """Tracking-first variant sidecar also loads correctly."""
    test_file = os.path.join('tests', 'data', '05_jv_extra.archive.yaml')
    entry_archive = parse(test_file)[0]

    assert isinstance(entry_archive.data, LabStabilityMeasurement)
    assert entry_archive.data.stability_tracking_file.endswith('(Tracking)_AI03-1A.txt')
    assert entry_archive.data.stability_parameters_file.endswith('(Parameters)_AI03-1A.txt')
    assert entry_archive.data.operator == 'Carol'
    assert entry_archive.data.pvk_sample is not None

