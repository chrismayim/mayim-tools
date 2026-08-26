"""Tests for the MayimManifest shared interface contract."""

import json


class TestMayimManifest:
    """Tests for MayimManifest creation and serialisation."""

    def test_manifest_can_be_created(self, tmp_path):
        """A manifest can be created with the required fields."""
        from mayim_tools.contract import MayimManifest

        raster_path = tmp_path / "test_dem.tif"
        raster_path.touch()

        manifest = MayimManifest.create(
            raster_path=str(raster_path),
            crs="EPSG:32735",
            cell_size=5.0,
            vertical_accuracy=0.15,
            nodata=-9999.0,
            produced_by="test-tool-0.2.0",
            stage=0,
        )

        assert manifest.raster_path == str(raster_path)
        assert manifest.crs == "EPSG:32735"
        assert manifest.cell_size == 5.0
        assert manifest.vertical_accuracy == 0.15
        assert manifest.nodata == -9999.0
        assert manifest.produced_by == "test-tool-0.2.0"
        assert manifest.has_parent is False

    def test_manifest_can_be_written_and_read(self, tmp_path):
        """A manifest survives a JSON write/read round trip."""
        from mayim_tools.contract import MayimManifest

        raster_path = tmp_path / "test_dem.tif"
        raster_path.touch()

        manifest_path = tmp_path / "test_dem.manifest.json"

        original = MayimManifest.create(
            raster_path=str(raster_path),
            crs="EPSG:4326",
            cell_size=0.0003,
            vertical_accuracy=4.0,
            nodata=-9999.0,
            produced_by="test-tool-0.2.0",
            stage=0,
        )

        original.write(str(manifest_path))

        restored = MayimManifest.read(str(manifest_path))

        assert restored.raster_path == original.raster_path
        assert restored.crs == original.crs
        assert restored.cell_size == original.cell_size
        assert restored.vertical_accuracy == original.vertical_accuracy
        assert restored.nodata == original.nodata
        assert restored.provenance_id == original.provenance_id
        assert restored.produced_by == original.produced_by

    def test_manifest_json_is_readable(self, tmp_path):
        """The manifest is written as indented, readable JSON."""
        from mayim_tools.contract import MayimManifest

        raster_path = tmp_path / "test_dem.tif"
        raster_path.touch()

        manifest_path = tmp_path / "test_dem.manifest.json"

        manifest = MayimManifest.create(
            raster_path=str(raster_path),
            crs="EPSG:32735",
            cell_size=5.0,
            vertical_accuracy=0.15,
            nodata=-9999.0,
            produced_by="test-tool-0.2.0",
        )

        manifest.write(str(manifest_path))

        with open(manifest_path, encoding="utf-8") as file:
            content = file.read()
            data = json.loads(content)

        assert "\n" in content
        assert "raster_path" in data
        assert "provenance_id" in data
