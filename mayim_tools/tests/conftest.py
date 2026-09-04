"""
Mayim Tools – pytest Configuration
Shared fixtures and setup for all Mayim Tools tests.
"""

import pytest


@pytest.fixture(scope="session")
def qgis_app():
    """
    Provide a QgsApplication instance for the test session.
    pytest-qgis handles this automatically — this fixture is here
    for explicitness and future customisation.
    """


@pytest.fixture
def sample_vector_layer(tmp_path):
    """
    Create a temporary in-memory vector layer for testing.
    Returns a valid QgsVectorLayer with a simple polygon feature.
    """
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsVectorLayer,
    )

    # Create an in-memory polygon layer
    layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "test_layer", "memory")
    assert layer.isValid(), "Test layer failed to create."

    # Add a simple square polygon feature
    feature = QgsFeature()
    feature.setGeometry(
        QgsGeometry.fromPolygonXY(
            [
                [
                    QgsPointXY(18.0, -34.0),
                    QgsPointXY(19.0, -34.0),
                    QgsPointXY(19.0, -33.0),
                    QgsPointXY(18.0, -33.0),
                    QgsPointXY(18.0, -34.0),
                ]
            ]
        )
    )
    layer.dataProvider().addFeature(feature)
    layer.updateExtents()

    return layer


@pytest.fixture
def sample_point_layer():
    """
    Create a temporary in-memory point layer for testing.
    Returns a valid QgsVectorLayer with a single point feature.
    """
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsVectorLayer,
    )

    layer = QgsVectorLayer("Point?crs=EPSG:4326", "test_points", "memory")
    assert layer.isValid(), "Test point layer failed to create."

    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(18.5, -33.5)))
    layer.dataProvider().addFeature(feature)
    layer.updateExtents()

    return layer
