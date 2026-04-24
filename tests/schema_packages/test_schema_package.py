import os.path

from nomad.client import normalize_all, parse


def test_schema_package():
    test_file = os.path.join('tests', 'data', 'test.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.message == 'Hello Markus!'

def test_cross_reference():
    base_dir = os.path.join('tests', 'data')

    files = os.path.join(base_dir, '02_jv_forward.archive.yaml')

    # Parse both files together (important!)
    archives = parse(files)

    # Normalize all together so references resolve
    for archive in archives:
        normalize_all(archive)

    # The second archive is your JV measurement
    jv_archive = archives[0]

    assert jv_archive.data.pvk_sample is not None
    assert jv_archive.data.pvk_sample.name == 'Test cell'