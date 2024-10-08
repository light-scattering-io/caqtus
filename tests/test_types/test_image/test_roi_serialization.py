import pytest

from caqtus.types.image.roi import RectangularROI, ROI, ArbitraryROI, converter


@pytest.mark.parametrize(
    "roi",
    [
        RectangularROI((100, 100), 50, 20, 10, 5),
        ArbitraryROI((100, 100), ((50, 20), (10, 5), (30, 40))),
    ],
)
def test_roi_serialization(roi):
    unstructured = converter.unstructure(roi, ROI)
    structured = converter.structure(unstructured, ROI)
    assert roi == structured
