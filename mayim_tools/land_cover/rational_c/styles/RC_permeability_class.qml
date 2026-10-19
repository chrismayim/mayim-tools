<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="4.0.3-Norrköping" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0">
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" band="1" nodataColor="" opacity="0.7" type="paletted">
      <colorPalette>
        <paletteEntry alpha="255" color="#2166ac" label="Very permeable" value="1"/>
        <paletteEntry alpha="255" color="#67a9cf" label="Very permeable / permeable" value="2"/>
        <paletteEntry alpha="255" color="#d1e5f0" label="Permeable" value="3"/>
        <paletteEntry alpha="255" color="#fddbc7" label="Permeable / semi-permeable" value="4"/>
        <paletteEntry alpha="255" color="#ef8a62" label="Semi-permeable" value="5"/>
        <paletteEntry alpha="255" color="#d6604d" label="Semi-permeable / impermeable" value="6"/>
        <paletteEntry alpha="255" color="#b2182b" label="Impermeable" value="7"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeStrength="100" grayscaleMode="0" colorizeBlue="128" colorizeRed="255" colorizeGreen="128" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
