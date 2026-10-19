<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="4.0.3-Norrköping" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0">
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" band="1" nodataColor="" opacity="0.7" type="paletted">
      <colorPalette>
        <paletteEntry alpha="255" color="#00CC00" label="Rural" value="1"/>
        <paletteEntry alpha="255" color="#FF0000" label="Urban" value="2"/>
        <paletteEntry alpha="255" color="#0000FF" label="Lakes and dams" value="3"/>
        <paletteEntry alpha="255" color="#7F7F7F" label="Ineffective (excluded)" value="4"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeStrength="100" grayscaleMode="0" colorizeBlue="128" colorizeRed="255" colorizeGreen="128" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
